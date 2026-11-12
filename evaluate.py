"""Score methods with PESQ-WB (16 kHz) and SNR, and log every run.

Scoring matches the course notebook: no level normalisation or time alignment,
plain mean over files. Clean speech is used only here, never in denoise.py.

    python evaluate.py                                     # submission config, validation
    python evaluate.py --split train --subset quick        # fast tuning set (~200 files)
    python evaluate.py --methods noisy_input basic_filter spectral_subtraction wiener_filter mmse_lsa
    python evaluate.py --config configs/xxx.json --set lsa_gain_floor=0.1 --tag floor01
    python evaluate.py --out results/submission            # refresh submitted-version results

Each run writes results/runs/<run_id>/ (per-file CSV, breakdowns, config) and
appends one line per method to results/experiments.csv.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from nr import DEFAULT, enhance, load_config  # noqa: E402

SUBMISSION = HERE / "configs" / "submission.json"
LOG = HERE / "results" / "experiments.csv"


def snr_db(clean, est):
    err = np.sum((clean - est) ** 2)
    return float(np.inf if err == 0 else 10 * np.log10(np.sum(clean ** 2) / err))


def pesq_wb(clean, est, sr):
    from pesq import PesqError, pesq
    try:
        return float(pesq(sr, clean, est, mode="wb", on_error=PesqError.RAISE_EXCEPTION))
    except Exception:
        return np.nan


def _score_one(job):
    root, row, methods, cfg = job
    clean, sr = sf.read(str(root / row["clean_path"]), dtype="float64")
    noisy, _ = sf.read(str(root / row["noisy_path"]), dtype="float64")
    out = []
    for m in methods:
        y = enhance(noisy, sr, m, cfg)
        out.append({"segment_id": row["segment_id"], "method": m,
                    "noise_category": row["noise_category"],
                    "noise_code": row["noise_code"],
                    "noise_difficulty": row["noise_difficulty"],
                    "target_snr_db": float(row["target_snr_db"]),
                    "pesq": pesq_wb(clean, y, sr), "snr_db": snr_db(clean, y)})
    return out


def parse_overrides(items):
    out = {}
    for item in items or []:
        k, v = item.split("=", 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def git_commit():
    try:
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE,
                           capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "nr", "configs",
                                "denoise.py", "evaluate.py"], cwd=HERE,
                               capture_output=True, text=True).stdout.strip()
        return h + ("+dirty" if dirty else "")
    except Exception:
        return "no-git"


def load_manifest(root: Path, split: str, subset: str | None) -> pd.DataFrame:
    m = pd.read_csv(root / split / "samples.csv", encoding="utf-8-sig")
    if subset:
        ids = (HERE / "splits" / f"{split}_{subset}.txt").read_text(encoding="utf-8").split()
        m = m[m.segment_id.isin(set(ids))]
    return m.reset_index(drop=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=HERE)
    ap.add_argument("--split", default="validation", choices=["train", "validation"])
    ap.add_argument("--subset", help="name of splits/<split>_<subset>.txt (e.g. quick)")
    ap.add_argument("--config", type=Path, help="JSON config (default: configs/submission.json)")
    ap.add_argument("--methods", nargs="+", help="methods to compare (default: config method)")
    ap.add_argument("--set", nargs="*", metavar="KEY=VALUE", help="override Config fields")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--limit", type=int, help="only the first N files (smoke test, not logged)")
    ap.add_argument("--tag", default="", help="short label for this run")
    ap.add_argument("--note", default="", help="free-text note saved in the experiment log")
    ap.add_argument("--out", type=Path, help="write results here instead of results/runs/<id>")
    args = ap.parse_args(argv)

    if args.config or not args.methods:
        cfg_method, cfg = load_config(args.config or SUBMISSION)
    else:
        cfg_method, cfg = None, DEFAULT
    cfg = cfg.with_overrides(**parse_overrides(args.set))
    methods = args.methods or [cfg_method]
    methods = ["noisy_input"] + [m for m in methods if m != "noisy_input"]

    root = args.data_root.resolve()
    manifest = load_manifest(root, args.split, args.subset)
    if args.limit:
        manifest = manifest.head(args.limit)
    jobs = [(root, r, methods, cfg) for r in manifest.to_dict("records")]

    t0 = time.time()
    with ProcessPoolExecutor(args.jobs) as ex:
        rows = [r for chunk in ex.map(_score_one, jobs, chunksize=4) for r in chunk]
    elapsed = time.time() - t0

    detail = pd.DataFrame(rows)
    base = detail[detail.method == "noisy_input"].set_index("segment_id")
    detail["delta_pesq"] = detail.pesq - detail.segment_id.map(base.pesq)
    detail["delta_snr_db"] = detail.snr_db - detail.segment_id.map(base.snr_db)

    order = {m: i for i, m in enumerate(methods)}
    summary = (detail.groupby("method")
               .agg(n=("pesq", "size"), pesq=("pesq", "mean"), pesq_std=("pesq", "std"),
                    pesq_fail=("pesq", lambda s: int(s.isna().sum())),
                    delta_pesq=("delta_pesq", "mean"), snr_db=("snr_db", "mean"),
                    delta_snr_db=("delta_snr_db", "mean"))
               .sort_index(key=lambda s: s.map(order)))

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + (f"_{args.tag}" if args.tag else "")
    out_dir = args.out or (HERE / "results" / "runs" / run_id)
    out_dir = out_dir if out_dir.is_absolute() else (Path.cwd() / out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(out_dir / "per_file.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "summary.csv", encoding="utf-8-sig")
    tables = {}
    for g in ("noise_difficulty", "target_snr_db", "noise_category"):
        tab = detail.pivot_table(index=g, columns="method", values="pesq", aggfunc="mean")
        tab = tab[[m for m in methods if m in tab.columns]]
        tab.insert(0, "n", detail[detail.method == "noisy_input"].groupby(g).size())
        tab.to_csv(out_dir / f"pesq_by_{g}.csv", encoding="utf-8-sig")
        tables[g] = tab
    commit = git_commit()
    run_info = {"run_id": run_id, "git_commit": commit, "split": args.split,
                "subset": args.subset or "all", "files": len(manifest), "methods": methods,
                "config": cfg.to_dict(), "note": args.note, "elapsed_sec": round(elapsed, 1),
                "pesq": "wb, 16 kHz, pesq==0.0.4, no normalisation",
                "aggregation": "unweighted mean over files"}
    (out_dir / "run_config.json").write_text(json.dumps(run_info, indent=2, ensure_ascii=False),
                                             encoding="utf-8")

    if not args.limit:  # smoke tests are not logged
        log_rows = []
        diff = tables["noise_difficulty"]
        snr = tables["target_snr_db"]
        for m, s in summary.iterrows():
            r = {"run_id": run_id, "git_commit": commit, "split": args.split,
                 "subset": args.subset or "all", "n": int(s.n), "method": m,
                 "pesq": round(s.pesq, 4), "delta_pesq": round(s.delta_pesq, 4),
                 "snr_db": round(s.snr_db, 3), "pesq_fail": int(s.pesq_fail)}
            for lvl in ("easy", "moderate", "hard"):
                r[f"pesq_{lvl}"] = round(diff.loc[lvl, m], 4) if lvl in diff.index else np.nan
            for db in (0.0, 5.0, 10.0, 15.0):
                r[f"pesq_{int(db)}dB"] = round(snr.loc[db, m], 4) if db in snr.index else np.nan
            r["config"] = json.dumps(cfg.to_dict()) if m != "noisy_input" else ""
            r["note"] = args.note
            log_rows.append(r)
        LOG.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(log_rows).to_csv(LOG, mode="a", header=not LOG.exists(), index=False,
                                      encoding="utf-8-sig" if not LOG.exists() else "utf-8")

    pd.set_option("display.width", 200)
    print(f"\n[{args.split}/{args.subset or 'all'}] {len(manifest)} files, {elapsed:.1f}s, "
          f"commit {commit}  -> {out_dir}")
    print(summary.round(3).to_string())
    for g in ("noise_difficulty", "target_snr_db"):
        print(f"\nPESQ by {g}\n" + tables[g].round(3).to_string())
    return summary


if __name__ == "__main__":
    main()
