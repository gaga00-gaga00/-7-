"""Submission rounds: record them, tag them, keep README's history table in sync.

    python tools/release.py prepare 1 --summary "MMSE-LSA ..."   # before the submission commit
    python tools/release.py tag 1                                # after committing
    python tools/release.py submitted 1 [--date 2026-11-13]      # after the Learn Us post
    python tools/release.py leaderboard 1 --pesq 1.70 [--snr 12.9]
    python tools/release.py readme                               # only re-render README table

prepare  checks results/submissions/roundN/ was scored with the current code
         (code fingerprint), then writes the round into submissions.json and
         README (team, round, tag, validation PESQ/SNR). Commit afterwards.
tag      requires a clean working tree and a matching fingerprint, creates the
         annotated tag submission-N on HEAD, and prints the push commands.
         Nothing is pushed by this script.
submitted / leaderboard   record the Learn Us submission date and hidden-test score.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.common import (CHALLENGE, DEADLINE, ROOT, TEAM, fingerprint, get_round, git,  # noqa: E402
                          load_submissions, round_dir, save_submissions, tag_name)

README = ROOT / "README.md"
START, END = "<!-- submissions:start -->", "<!-- submissions:end -->"


def round_result(n: int) -> dict:
    """Validation summary of round n's submitted method (not the noisy input)."""
    d = round_dir(n)
    info = json.loads((d / "run_config.json").read_text(encoding="utf-8"))
    with open(d / "summary.csv", encoding="utf-8-sig") as f:
        rows = {r["method"]: r for r in csv.DictReader(f)}
    method = [m for m in info["methods"] if m != "noisy_input"][-1]
    return {"method": method, "files": info["files"],
            "pesq": round(float(rows[method]["pesq"]), 4),
            "snr_db": round(float(rows[method]["snr_db"]), 3),
            "noisy_pesq": round(float(rows["noisy_input"]["pesq"]), 4),
            "fingerprint": info.get("code_fingerprint")}


def require_fingerprint(n: int) -> dict:
    res = round_result(n)
    now = fingerprint()
    if res["fingerprint"] != now:
        sys.exit(f"results/submissions/round{n}/ was scored with code {res['fingerprint']}, "
                 f"current code is {now}.\nRe-run:  python evaluate.py --round {n}")
    return res


def render_table(data: dict) -> str:
    lines = [f"팀명: **{data['team']}** · 저장소: {data['repository']}", "",
             "| 회차 | 태그 | 커밋 | 런어스 제출일 | 방법 | Validation PESQ | Validation SNR (dB) "
             "| 히든 테스트 PESQ |",
             "|---|---|---|---|---|---|---|---|"]
    for r in sorted(data["rounds"], key=lambda r: r["round"]):
        v = r.get("validation", {})
        lb = r.get("leaderboard") or {}
        lines.append("| {round} | `{tag}` | {commit} | {sub} | {method} | {pesq} | {snr} | {lb} |".format(
            round=r["round"], tag=r["tag"],
            commit=f"`{r['commit'][:7]}`" if r.get("commit") else "태그 참조",
            sub=r.get("submitted_at") or "-", method=r.get("method_summary") or v.get("method", ""),
            pesq=f"{v['pesq']:.3f}" if v else "-", snr=f"{v['snr_db']:.2f}" if v else "-",
            lb=f"{lb['pesq']:.3f}" if lb.get("pesq") is not None else "-"))
    lines += ["", "평가할 코드 버전 = 해당 회차 **태그**가 가리키는 커밋 "
              "(`git checkout <태그>`). 회차별 Validation 결과: "
              "[`results/submissions/`](results/submissions/)"]
    return "\n".join(lines)


def write_readme(data: dict) -> None:
    text = README.read_text(encoding="utf-8")
    if START not in text or END not in text:
        sys.exit(f"README.md has no {START} ... {END} block")
    a, b = text.index(START) + len(START), text.index(END)
    README.write_text(text[:a] + "\n" + render_table(data) + "\n" + text[b:], encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare"); p.add_argument("round", type=int)
    p.add_argument("--summary", help="one-line description of the method for the table")
    p = sub.add_parser("tag"); p.add_argument("round", type=int)
    p = sub.add_parser("submitted"); p.add_argument("round", type=int)
    p.add_argument("--date", default=date.today().isoformat())
    p = sub.add_parser("leaderboard"); p.add_argument("round", type=int)
    p.add_argument("--pesq", type=float, required=True); p.add_argument("--snr", type=float)
    sub.add_parser("readme")
    args = ap.parse_args(argv)

    data = load_submissions()
    data.setdefault("challenge", CHALLENGE)
    data.setdefault("deadline", DEADLINE)
    data["team"] = data.get("team") or TEAM

    if args.cmd == "prepare":
        res = require_fingerprint(args.round)
        r = get_round(data, args.round)
        if r is None:
            r = {"round": args.round}
            data["rounds"].append(r)
        r.update({"tag": tag_name(args.round), "config": "configs/submission.json",
                  "results": f"results/submissions/round{args.round}/",
                  "validation": {k: res[k] for k in ("method", "files", "pesq", "snr_db")},
                  "code_fingerprint": res["fingerprint"]})
        if args.summary:
            r["method_summary"] = args.summary
        r.setdefault("commit", None)
        r.setdefault("submitted_at", None)
        r.setdefault("leaderboard", None)
        save_submissions(data)
        write_readme(data)
        print(f"round {args.round} recorded: {res['method']} PESQ {res['pesq']:.4f} "
              f"SNR {res['snr_db']:.2f} dB\nNext: commit, then  python tools/release.py tag "
              f"{args.round}")

    elif args.cmd == "tag":
        r = get_round(data, args.round)
        if r is None:
            sys.exit(f"round {args.round} not in submissions.json; run prepare first")
        require_fingerprint(args.round)
        if git("status", "--porcelain"):
            sys.exit("working tree is not clean: commit first (the tag must match what is pushed)")
        tag = tag_name(args.round)
        if git("tag", "--list", tag):
            sys.exit(f"tag {tag} already exists ({git('rev-list', '-n1', tag)[:7]})")
        head = git("rev-parse", "HEAD")
        git("tag", "-a", tag, "-m", f"{TEAM} submission round {args.round}")
        print(f"created {tag} -> {head[:7]}\nPush (after checking):\n"
              f"  git push origin main\n  git push origin {tag}\n"
              f"Then record the commit:  python tools/release.py submitted {args.round}")

    elif args.cmd == "submitted":
        r = get_round(data, args.round)
        tag = tag_name(args.round)
        if r is None or not git("tag", "--list", tag):
            sys.exit(f"round {args.round} has no tag {tag} yet")
        r["commit"] = git("rev-list", "-n1", tag)
        r["submitted_at"] = args.date
        save_submissions(data)
        write_readme(data)
        print(f"round {args.round}: commit {r['commit'][:7]}, submitted {args.date}")

    elif args.cmd == "leaderboard":
        r = get_round(data, args.round)
        if r is None:
            sys.exit(f"round {args.round} not in submissions.json")
        r["leaderboard"] = {"pesq": args.pesq, "snr_db": args.snr}
        save_submissions(data)
        write_readme(data)
        print(f"round {args.round}: hidden-test PESQ {args.pesq}")

    else:
        save_submissions(data)
        write_readme(data)
        print("README submissions table updated")


if __name__ == "__main__":
    main()
