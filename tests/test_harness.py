"""Harness tests: pass-through fallback, code fingerprint, submissions table, AI report.

    python -m pytest -q
"""
from __future__ import annotations

import json
import shutil
import sys
import zipfile
from pathlib import Path
from xml.dom import minidom

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import denoise  # noqa: E402
from tools import build_ai_usage, release  # noqa: E402
from tools.common import fingerprint, tag_name  # noqa: E402


def test_failed_file_is_copied_unprocessed(tmp_path, monkeypatch):
    inp, out = tmp_path / "input", tmp_path / "output"
    inp.mkdir()
    rng = np.random.default_rng(0)
    for name in ("good.wav", "bad.wav"):
        sf.write(str(inp / name), 0.1 * rng.standard_normal(16000), 16000, subtype="PCM_16")
    real = denoise.enhance

    def flaky(x, sr, method, cfg):
        if flaky.calls == 0:
            flaky.calls += 1
            raise RuntimeError("boom")
        return real(x, sr, method, cfg)
    flaky.calls = 0
    monkeypatch.setattr(denoise, "enhance", flaky)
    assert denoise.main(["--input", str(inp), "--output", str(out)]) == 0
    assert sorted(p.name for p in out.iterdir()) == ["bad.wav", "good.wav"]
    # sorted order: bad.wav is processed first and fails -> byte-identical copy
    assert (out / "bad.wav").read_bytes() == (inp / "bad.wav").read_bytes()


def _copy_code(dst: Path):
    for rel in ["denoise.py", "requirements.txt", "configs/submission.json",
                *[p.relative_to(ROOT).as_posix() for p in (ROOT / "nr").glob("*.py")]]:
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, dst / rel)


def test_fingerprint_tracks_code_not_line_endings(tmp_path):
    _copy_code(tmp_path)
    assert fingerprint(tmp_path) == fingerprint(ROOT)
    cfg = tmp_path / "configs" / "submission.json"
    cfg.write_bytes(cfg.read_bytes().replace(b"\n", b"\r\n"))
    assert fingerprint(tmp_path) == fingerprint(ROOT)
    data = json.loads(cfg.read_text(encoding="utf-8"))
    data["params"]["lsa_gain_floor"] = 0.123
    cfg.write_text(json.dumps(data), encoding="utf-8")
    assert fingerprint(tmp_path) != fingerprint(ROOT)


def test_submissions_table():
    data = {"team": "7조", "repository": "https://example.org/repo", "rounds": [
        {"round": 2, "tag": tag_name(2), "commit": None, "submitted_at": None,
         "validation": {"method": "x", "pesq": 1.8, "snr_db": 13.5}, "leaderboard": None},
        {"round": 1, "tag": tag_name(1), "commit": "abcdef1234", "submitted_at": "2026-11-13",
         "validation": {"method": "mmse_lsa", "pesq": 1.7177, "snr_db": 13.29},
         "leaderboard": {"pesq": 1.65, "snr_db": None}, "method_summary": "MMSE-LSA"}]}
    lines = release.render_table(data).split("\n")
    rows = [ln for ln in lines if ln.startswith("| 1 ") or ln.startswith("| 2 ")]
    assert rows[0].startswith("| 1 | `submission-1` | `abcdef1` | 2026-11-13 | MMSE-LSA | 1.718")
    assert rows[0].endswith("| 1.650 |")
    assert rows[1].startswith("| 2 | `submission-2` | 태그 참조 | - |")
    assert "7조" in lines[0]


def test_readme_has_submissions_block():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert release.START in text and release.END in text


SPEC = {
    "output": "x.hwpx", "team": "7", "members": "가, 나", "models": ["Claude"],
    "purposes": ["concept"],
    "cases": [{"title": "t", "prompt": ["p"], "answer": ["a"], "verify": ["v"]}],
    "algorithm": ["## 원리", "설명"], "references": [], "analysis": ["분석"],
    "reflection": ["고찰"],
    "results": {"rows": [], "notes": []},
}


def test_validate_spec_reports_missing_items():
    problems = build_ai_usage.validate_spec({**SPEC, "analysis": ["TODO 청취"]})
    text = "\n".join(problems)
    assert "references 없음" in text
    assert "results.rows" in text
    assert "TODO" in text


@pytest.mark.skipif(not (ROOT / "docs" / "ai_usage" / "template.hwpx").exists(),
                    reason="school form is not in the repository")
def test_build_ai_usage_produces_valid_hwpx(tmp_path):
    out = build_ai_usage.build(SPEC, tmp_path / "x.hwpx")
    with zipfile.ZipFile(out) as z:
        xml = z.read("Contents/section0.xml").decode("utf-8")
        assert z.infolist()[0].filename == "mimetype"
    minidom.parseString(xml)
    for title in ("Part 3. 알고리즘 이해", "Part 4. 참고 문헌", "Part 5. 개선 결과",
                  "Part 6. 분석 및 고찰", "Part 7. 종합 고찰"):
        assert title in xml
    assert "사례1: t" in xml and "(  7  ) 조" in xml


@pytest.mark.parametrize("argv", [
    ["--split", "all", "--subset", "quick"],
    ["--round", "1", "--split", "train"],
    ["--round", "1", "--set", "lsa_gain_floor=0.1"],
])
def test_evaluate_rejects_partial_round_or_subset_of_all(argv):
    import evaluate
    with pytest.raises(SystemExit):
        evaluate.main(argv)


def test_ai_usage_reads_round_train_results():
    if not (ROOT / "results" / "submissions" / "round1" / "train" / "summary.csv").exists():
        pytest.skip("round 1 train results not generated")
    v = build_ai_usage.result_value("round1/train", "mmse_lsa")
    assert v["n"] == 1296
