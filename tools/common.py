"""Shared constants and helpers for the harness tools (not needed by denoise.py)."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CHALLENGE = "2026 신호 및 시스템 1차 챌린지 - 전통적 신호 처리 기반 음성 잡음 제거"
DEADLINE = "2026-11-17 23:59"
TEAM = "7조"
REPO_URL = "https://github.com/gaga00-gaga00/-7-"
TAG_PREFIX = "submission-"

SUBMISSIONS = ROOT / "submissions.json"
RESULTS = ROOT / "results" / "submissions"
AI_USAGE = ROOT / "docs" / "ai_usage"
REFERENCES = ROOT / "docs" / "references.json"

# Files whose content defines what denoise.py outputs. If any changes, the
# validation results saved for a submission no longer describe the code.
FINGERPRINT_GLOBS = ["denoise.py", "nr/*.py", "configs/submission.json", "requirements.txt"]


def tag_name(round_no: int) -> str:
    return f"{TAG_PREFIX}{round_no}"


def round_dir(round_no: int) -> Path:
    return RESULTS / f"round{round_no}"


def fingerprint(root: Path = ROOT) -> str:
    """Short hash of the submission code (line endings normalised)."""
    h = hashlib.sha256()
    files = sorted({p for g in FINGERPRINT_GLOBS for p in root.glob(g) if p.is_file()})
    for p in files:
        h.update(p.relative_to(root).as_posix().encode())
        h.update(b"\0")
        h.update(p.read_bytes().replace(b"\r\n", b"\n"))
        h.update(b"\0")
    return h.hexdigest()[:12]


def git(*args, cwd: Path = ROOT, check: bool = True) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True,
                          check=check, encoding="utf-8").stdout.strip()


def load_submissions() -> dict:
    if SUBMISSIONS.exists():
        return json.loads(SUBMISSIONS.read_text(encoding="utf-8"))
    return {"team": TEAM, "repository": REPO_URL, "rounds": []}


def save_submissions(data: dict) -> None:
    SUBMISSIONS.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")


def get_round(data: dict, round_no: int) -> dict | None:
    return next((r for r in data["rounds"] if r["round"] == round_no), None)
