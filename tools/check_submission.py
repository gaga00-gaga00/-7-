"""Pre-submission check: the challenge's "제출 전 확인" list, item by item.

    python tools/check_submission.py --round 1 [--python "py -3.13"] [--n 0] [--keep]

What it does, in the TA's place:
  1. requirements.txt      fresh venv, `pip install -r requirements.txt` only
  2. 실행 방법             README's commands, `python denoise.py` in the repo root
  3. input -> output       every validation noisy WAV gets exactly one output
  4. 형식 유지             sample rate, channels, frames, subtype of every output
     (+ re-scores the outputs and compares with results/submissions/roundN/)
  5. GitHub / 제출 버전    anonymous access to the public repo, tag submission-N
                           pushed and equal to the local tag, README/submissions.json
  6. AI 활용 내역서        docs/ai_usage/roundN.json complete, references checked,
                           AI활용내역서_N차.hwpx built
  7. 제출 횟수             at most one submission per week (first one not counted)

Code source: the tag submission-N if it exists (exactly what the TA will clone),
otherwise the working tree as it would be committed (`git add -A`), with a warning.
Exit code 0 = no FAIL. Needs network for pip and GitHub.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.common import (AI_USAGE, REPO_URL, fingerprint, get_round, git,  # noqa: E402
                          load_submissions, round_dir, tag_name)

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


class Report:
    def __init__(self):
        self.items: list[tuple[str, str, str]] = []

    def add(self, item: str, status: str, detail: str = ""):
        self.items.append((item, status, detail))
        print(f"[{status}] {item}" + (f" — {detail}" if detail else ""), flush=True)

    def print(self):
        print("\n" + "=" * 72)
        for item, status, detail in self.items:
            print(f"[{status}] {item}")
            if detail and status != PASS:
                for line in detail.split("\n"):
                    print(f"         {line}")
        fails = sum(s == FAIL for _, s, _ in self.items)
        warns = sum(s == WARN for _, s, _ in self.items)
        print("=" * 72)
        print("READY TO SUBMIT" if not fails else "NOT READY", f"(FAIL {fails}, WARN {warns})")
        return fails


def sh(cmd, cwd=None, env=None):
    print("$", " ".join(map(str, cmd)), flush=True)
    return subprocess.run(cmd, cwd=cwd, env=env, check=True, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")


def checkout_code(tag: str, dest: Path, rep: Report) -> str:
    """Put the code the TA will see into dest; return a description."""
    if git("tag", "--list", tag):
        sh(["git", "clone", "-q", "--no-local", str(ROOT), str(dest)])
        sh(["git", "checkout", "-q", tag], cwd=dest)
        return f"tag {tag} ({git('rev-list', '-n1', tag)[:7]})"
    files = git("ls-files", "--cached", "--others", "--exclude-standard").split("\n")
    for rel in filter(None, files):
        if (ROOT / rel).is_file():
            (dest / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / rel, dest / rel)
    dirty = git("status", "--porcelain")
    rep.add("코드 출처", WARN, f"태그 {tag} 없음 → 작업 트리(커밋 예정 상태)로 점검"
            + (" · 커밋 안 된 변경 있음" if dirty else ""))
    return "working tree"


def check_formats(noisy: list[Path], out_dir: Path, rep: Report, before: set[str]):
    import soundfile as sf
    produced = sorted(p.relative_to(out_dir).as_posix() for p in out_dir.rglob("*")
                      if p.is_file()) if out_dir.is_dir() else []
    produced = [p for p in produced if p not in before]  # e.g. output/.gitkeep
    expected = sorted(f.name for f in noisy)
    missing = sorted(set(expected) - set(produced))
    extra = sorted(set(produced) - set(expected))
    rep.add("3. input의 모든 파일 → output에 같은 이름으로 저장",
            PASS if not missing and not extra else FAIL,
            f"{len(expected)}개 입력, {len(produced)}개 출력"
            + (f"\n누락: {missing[:5]}" if missing else "") + (f"\n불필요: {extra[:5]}" if extra else ""))
    bad = []
    for f in noisy:
        o = out_dir / f.name
        if o.exists():
            a, b = sf.info(str(f)), sf.info(str(o))
            if (a.samplerate, a.channels, a.frames, a.subtype) != \
               (b.samplerate, b.channels, b.frames, b.subtype):
                bad.append(f"{f.name}: {a.samplerate}/{a.channels}/{a.frames}/{a.subtype} -> "
                           f"{b.samplerate}/{b.channels}/{b.frames}/{b.subtype}")
    rep.add("4. 샘플링 주파수·채널 수·샘플 수(·포맷) 유지", PASS if not bad else FAIL,
            f"불일치 {len(bad)}개" + ("\n" + "\n".join(bad[:5]) if bad else ""))


def rescore(noisy: list[Path], out_dir: Path, round_no: int, rep: Report):
    """Score the files denoise.py wrote and compare with the saved round results."""
    item = "4+. 제출 코드 출력 점수 = 저장된 Validation 결과"
    summary = round_dir(round_no) / "summary.csv"
    if not summary.exists():
        rep.add(item, FAIL, f"{summary.relative_to(ROOT)} 없음 → python evaluate.py --round {round_no}")
        return
    try:
        import soundfile as sf
        from evaluate import pesq_wb, snr_db
        import pesq  # noqa: F401
    except ImportError as e:
        rep.add(item, WARN, f"개발 환경에 {e.name} 없음 → 점수 비교 생략")
        return
    with open(ROOT / "validation" / "samples.csv", encoding="utf-8-sig") as f:
        clean_of = {Path(r["noisy_path"]).name: ROOT / r["clean_path"] for r in csv.DictReader(f)}
    ps, ss = [], []
    for f in noisy:
        clean, sr = sf.read(str(clean_of[f.name]), dtype="float64")
        y, _ = sf.read(str(out_dir / f.name), dtype="float64")
        ps.append(pesq_wb(clean, y, sr))
        ss.append(snr_db(clean, y))
    import numpy as np
    pesq_out, snr_out = float(np.nanmean(ps)), float(np.mean(ss))
    with open(summary, encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r["method"] != "noisy_input"]
    saved_pesq, saved_snr = float(rows[-1]["pesq"]), float(rows[-1]["snr_db"])
    n_saved = int(rows[-1]["n"])
    detail = (f"출력 WAV 재채점 PESQ {pesq_out:.4f} / SNR {snr_out:.2f} dB, "
              f"저장된 결과 PESQ {saved_pesq:.4f} / SNR {saved_snr:.2f} dB")
    if len(noisy) != n_saved:
        rep.add(item, WARN, detail + f" (일부 {len(noisy)}/{n_saved}개만 비교 — 전체는 --n 0)")
    else:
        ok = abs(pesq_out - saved_pesq) < 0.005 and abs(snr_out - saved_snr) < 0.05
        rep.add(item, PASS if ok else FAIL, detail)


def check_remote(round_no: int, rep: Report):
    tag = tag_name(round_no)
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "never"}
    try:
        out = subprocess.run(["git", "-c", "credential.helper=", "ls-remote", REPO_URL,
                              "refs/heads/main", f"refs/tags/{tag}", f"refs/tags/{tag}^{{}}"],
                             capture_output=True, text=True, env=env, timeout=60)
    except Exception as e:
        rep.add("5a. GitHub 저장소 공개 접근", FAIL, f"{type(e).__name__}: {e}")
        return
    if out.returncode != 0:
        rep.add("5a. GitHub 저장소 공개 접근", FAIL,
                f"로그인 없이 접근 불가 (비공개?): {out.stderr.strip()[:200]}")
        return
    rep.add("5a. GitHub 저장소 공개 접근", PASS, REPO_URL)
    refs = dict(reversed(line.split("\t")) for line in out.stdout.strip().split("\n") if line)
    remote = refs.get(f"refs/tags/{tag}^{{}}") or refs.get(f"refs/tags/{tag}")
    local = git("rev-list", "-n1", tag, check=False) if git("tag", "--list", tag) else None
    if not local:
        rep.add(f"5b. 제출 버전 태그 {tag}", FAIL,
                f"로컬 태그 없음 → 커밋 후 python tools/release.py tag {round_no}")
    elif not remote:
        rep.add(f"5b. 제출 버전 태그 {tag}", FAIL, f"GitHub에 없음 → git push origin {tag}")
    elif remote != local:
        rep.add(f"5b. 제출 버전 태그 {tag}", FAIL,
                f"GitHub {remote[:7]} ≠ 로컬 {local[:7]}")
    else:
        rep.add(f"5b. 제출 버전 태그 {tag}", PASS, f"GitHub = 로컬 = {local[:7]}")


def check_version_info(round_no: int, code_dir: Path, rep: Report):
    tag = tag_name(round_no)
    readme = (code_dir / "README.md").read_text(encoding="utf-8")
    sub_path = code_dir / "submissions.json"
    sub = json.loads(sub_path.read_text(encoding="utf-8")) if sub_path.exists() else {"rounds": []}
    r = get_round(sub, round_no)
    problems = []
    if r is None:
        problems.append(f"submissions.json에 {round_no}회차 없음 → tools/release.py prepare {round_no}")
    if sub.get("team", "") not in readme or not sub.get("team"):
        problems.append("README에 팀명 없음")
    if f"`{tag}`" not in readme:
        problems.append(f"README 제출 정보 표에 {tag} 없음")
    rep.add("5c. README에 팀명·제출 회차·태그 명시", PASS if not problems else FAIL,
            "\n".join(problems))
    fp_code = fingerprint(code_dir)
    item = "5d. 저장된 Validation·Train 전체 결과가 제출 코드로 만든 것인가"
    problems = []
    for split, folder in (("validation", round_dir(round_no)),
                          ("train", round_dir(round_no) / "train")):
        saved = folder / "run_config.json"
        if not saved.exists():
            problems.append(f"{split}: {folder.relative_to(ROOT)} 없음")
            continue
        info = json.loads(saved.read_text(encoding="utf-8"))
        if info.get("code_fingerprint") != fp_code:
            problems.append(f"{split}: 결과 {info.get('code_fingerprint')} / 코드 {fp_code}")
        expected = len(list((ROOT / split / "noisy").glob("*.wav")))
        if info.get("subset") != "all" or (expected and info.get("files") != expected):
            problems.append(f"{split}: {info.get('files')}개만 평가됨 (전체 {expected}개)")
    rep.add(item, PASS if not problems else FAIL,
            f"코드 {fp_code}" if not problems else
            "\n".join(problems) + f"\n→ python evaluate.py --round {round_no}")


def check_ai_usage(round_no: int, rep: Report):
    spec_path = AI_USAGE / f"round{round_no}.json"
    if not spec_path.exists():
        rep.add("6. AI 활용 내역서·참고 문헌", FAIL, f"{spec_path.relative_to(ROOT)} 없음")
        return
    from tools.build_ai_usage import validate_spec
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    problems = validate_spec(spec)
    hwpx = AI_USAGE / spec.get("output", "")
    if not hwpx.is_file():
        problems.append(f"{spec.get('output')} 미생성 → python tools/build_ai_usage.py "
                        f"docs/ai_usage/round{round_no}.json")
    elif hwpx.stat().st_mtime < spec_path.stat().st_mtime:
        problems.append(f"{hwpx.name}가 round{round_no}.json보다 오래됨 → 다시 생성")
    rep.add("6. AI 활용 내역서·참고 문헌", PASS if not problems else FAIL, "\n".join(problems))


def check_weekly(round_no: int, rep: Report):
    sub = load_submissions()
    done = sorted((r for r in sub["rounds"] if r.get("submitted_at") and r["round"] != round_no),
                  key=lambda r: r["round"])
    if not done:
        rep.add("7. 제출 횟수 (주 1회, 최초 제출 제외)", PASS, "최초 제출")
        return
    week = date.today().isocalendar()[:2]
    counted = [r for r in done[1:] if date.fromisoformat(r["submitted_at"]).isocalendar()[:2] == week]
    rep.add("7. 제출 횟수 (주 1회, 최초 제출 제외)", PASS if not counted else WARN,
            "" if not counted else
            f"이번 주에 이미 {counted[0]['round']}회차 제출 — 막바지 완화 공지가 없으면 다음 주에")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--round", type=int, required=True, help="submission round to check")
    ap.add_argument("--n", type=int, default=0, help="validation files to run (0 = all)")
    ap.add_argument("--python", default=sys.executable,
                    help='interpreter for the venv, e.g. "py -3.13"')
    ap.add_argument("--skip-github", action="store_true", help="skip item 5a/5b (offline)")
    ap.add_argument("--keep", action="store_true", help="keep the temp folder")
    args = ap.parse_args(argv)

    rep = Report()
    tmp = Path(tempfile.mkdtemp(prefix="nr_submit_"))
    repo = tmp / "repo"
    repo.mkdir()
    try:
        source = checkout_code(tag_name(args.round), repo, rep)
        print(f"code: {source}")

        venv = tmp / "venv"
        try:
            sh([*shlex.split(args.python), "-m", "venv", str(venv)])
            py = venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
            sh([str(py), "-m", "pip", "install", "-q", "--disable-pip-version-check",
                "-r", "requirements.txt"], cwd=repo)
            ver = sh([str(py), "-c", "import sys,numpy,scipy,soundfile;print(sys.version.split()[0],"
                      "numpy.__version__,scipy.__version__,soundfile.__version__)"]).stdout.split()
            rep.add("1. requirements.txt로 패키지 설치", PASS,
                    f"Python {ver[0]}: numpy {ver[1]}, scipy {ver[2]}, soundfile {ver[3]}")
        except subprocess.CalledProcessError as e:
            rep.add("1. requirements.txt로 패키지 설치", FAIL, (e.stderr or e.stdout)[-800:])
            return rep.print()

        readme = (repo / "README.md").read_text(encoding="utf-8")
        noisy = sorted((ROOT / "validation" / "noisy").glob("*.wav"))
        noisy = noisy[: args.n] if args.n else noisy
        (repo / "input").mkdir(exist_ok=True)
        for f in noisy:
            shutil.copy2(f, repo / "input" / f.name)
        out_dir = repo / "output"
        before = {p.relative_to(out_dir).as_posix() for p in out_dir.rglob("*")
                  if p.is_file()} if out_dir.is_dir() else set()
        documented = all(c in readme for c in ("pip install -r requirements.txt", "python denoise.py"))
        try:
            run = sh([str(py), "denoise.py"], cwd=repo)
            warn = [ln for ln in run.stderr.splitlines() if ln.startswith(("[WARN]", "[FAIL]"))]
            status = PASS if documented and not warn else (WARN if documented else FAIL)
            rep.add("2. 안내한 실행 방법대로 동작", status,
                    run.stdout.strip().splitlines()[-1]
                    + ("" if documented else "\nREADME에 설치·실행 명령이 없음")
                    + ("\n" + "\n".join(warn[:5]) if warn else ""))
        except subprocess.CalledProcessError as e:
            rep.add("2. 안내한 실행 방법대로 동작", FAIL, (e.stderr or e.stdout)[-800:])
            return rep.print()

        check_formats(noisy, out_dir, rep, before)
        rescore(noisy, repo / "output", args.round, rep)
        if args.skip_github:
            rep.add("5a/5b. GitHub 공개 접근·태그", WARN, "--skip-github 로 생략")
        else:
            check_remote(args.round, rep)
        check_version_info(args.round, repo, rep)
        check_ai_usage(args.round, rep)
        check_weekly(args.round, rep)
        return rep.print()
    finally:
        if args.keep:
            print("kept:", tmp)
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(1 if main() else 0)
