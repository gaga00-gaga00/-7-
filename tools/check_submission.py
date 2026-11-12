"""Simulate the TA's run: fresh venv + requirements.txt only + git-tracked files.

1. Copies the files git tracks (what the TA will clone) into a temp folder.
2. Creates a fresh venv there and installs ONLY requirements.txt.
3. Copies N validation noisy WAVs to input/, runs `python denoise.py`.
4. Checks every output: same name, sample rate, channels, frames, format.

    python tools/check_submission.py [--n 20] [--python py -3.13] [--keep]

Needs network access for pip. Exit code 0 = ready to submit.
"""
from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sh(cmd, cwd=None):
    print("$", " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--python", default=sys.executable,
                    help='interpreter for the venv, e.g. "py -3.13"')
    ap.add_argument("--keep", action="store_true", help="keep the temp folder")
    args = ap.parse_args(argv)

    tracked = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT, capture_output=True, text=True,
                             check=True).stdout.split("\n")
    tracked = [t for t in tracked if t]
    if "denoise.py" not in tracked:
        sys.exit("denoise.py would not be committed (check .gitignore).")

    tmp = Path(tempfile.mkdtemp(prefix="nr_submit_"))
    repo = tmp / "repo"
    try:
        for rel in tracked:
            dst = repo / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / rel, dst)
        venv = tmp / "venv"
        sh([*shlex.split(args.python), "-m", "venv", str(venv)])
        py = venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        sh([str(py), "-m", "pip", "install", "-q", "--disable-pip-version-check",
            "-r", str(repo / "requirements.txt")])

        import soundfile as sf  # the checker itself runs in the dev env
        noisy = sorted((ROOT / "validation" / "noisy").glob("*.wav"))[: args.n]
        (repo / "input").mkdir(exist_ok=True)
        for f in noisy:
            shutil.copy2(f, repo / "input" / f.name)
        sh([str(py), "denoise.py"], cwd=repo)

        bad = []
        for f in noisy:
            o = repo / "output" / f.name
            if not o.exists():
                bad.append(f"{f.name}: missing")
                continue
            a, b = sf.info(str(f)), sf.info(str(o))
            if (a.samplerate, a.channels, a.frames, a.subtype) != \
               (b.samplerate, b.channels, b.frames, b.subtype):
                bad.append(f"{f.name}: {a} vs {b}")
        print(f"\nchecked {len(noisy)} files, problems: {len(bad)}")
        for b in bad:
            print("  ", b)
        print("READY TO SUBMIT" if not bad else "NOT READY")
        return 0 if not bad else 1
    finally:
        if args.keep:
            print("kept:", tmp)
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
