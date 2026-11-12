"""Check that every reference in docs/references.json really exists (Crossref).

    python tools/check_refs.py            # look up each DOI, compare title and year
    python tools/check_refs.py --write    # also record doi_verified (date) when they match

Only the DOI is sent to api.crossref.org. This confirms the paper exists with
that title/year; reading it and filling "checked_by" is still the team's job
(the notice: don't copy AI-suggested lists without checking them yourself).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCES = ROOT / "docs" / "references.json"


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def lookup(doi: str) -> dict:
    req = urllib.request.Request(f"https://api.crossref.org/works/{doi}",
                                 headers={"User-Agent": "reference-checker/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        msg = json.load(r)["message"]
    year = (msg.get("issued", {}).get("date-parts") or [[None]])[0][0]
    return {"title": (msg.get("title") or [""])[0], "year": year,
            "venue": (msg.get("container-title") or [""])[0]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)
    refs = json.loads(REFERENCES.read_text(encoding="utf-8"))
    bad = 0
    for r in refs:
        if not r.get("doi"):
            print(f"[--]   {r['id']}: DOI 없음 — URL을 직접 열어 확인하세요 ({r.get('url')})")
            continue
        try:
            found = lookup(r["doi"])
        except Exception as e:
            print(f"[FAIL] {r['id']}: {r['doi']} 조회 실패 ({type(e).__name__}: {e})")
            bad += 1
            continue
        ok = norm(found["title"]) == norm(r["title"]) and found["year"] == r["year"]
        print(f"[{'OK' if ok else 'DIFF'}]   {r['id']}: {found['title']} ({found['year']}, "
              f"{found['venue']})")
        if not ok:
            print(f"         기재: {r['title']} ({r['year']})")
            bad += 1
        elif args.write:
            r["doi_verified"] = date.today().isoformat()
        if not r.get("checked_by"):
            print(f"         checked_by 비어 있음 — 원문 확인 후 확인한 팀원 이름을 적으세요")
    if args.write:
        REFERENCES.write_text(json.dumps(refs, indent=2, ensure_ascii=False) + "\n",
                              encoding="utf-8")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
