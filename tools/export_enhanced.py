"""Save enhanced WAVs for a whole split (course notebook's export_enhanced).

    python tools/export_enhanced.py                                   # submission config, validation
    python tools/export_enhanced.py --methods basic_filter spectral_subtraction wiener_filter mmse_lsa
    python tools/export_enhanced.py --split train --subset quick --config configs/X.json

Writes results/enhanced/<split>/<method>/<noisy file name> as FLOAT WAV (same
sample rate and length, no clipping or normalisation), plus <split>_outputs.csv
(path, peak) and config.json. Only noisy inputs are read. Git-ignored.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluate import load_manifest, read_checked  # noqa: E402
from nr import enhance, load_config  # noqa: E402

SUBMISSION = ROOT / "configs" / "submission.json"


def _export_one(job):
    row, methods, cfg, dest = job
    noisy, sr = read_checked(ROOT, row, "noisy")
    records = []
    for m in methods:
        y = enhance(noisy, sr, m, cfg)
        target = dest / m / Path(row["noisy_path"]).name
        target.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(target), y, sr, subtype="FLOAT")
        records.append({"segment_id": row["segment_id"], "method": m,
                        "output_path": target.relative_to(ROOT).as_posix(),
                        "peak": float(np.abs(y).max()) if y.size else 0.0})
    return records


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="validation", choices=["train", "validation"])
    ap.add_argument("--subset", help="splits/<split>_<subset>.txt")
    ap.add_argument("--config", type=Path, default=SUBMISSION)
    ap.add_argument("--methods", nargs="+", help="default: the config's method")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "enhanced")
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args(argv)

    method, cfg = load_config(args.config)
    methods = args.methods or [method]
    out = args.out.resolve()
    for split in ("train", "validation"):  # never write next to the original audio
        if out.is_relative_to(ROOT / split):
            ap.error("choose an output folder outside train/ and validation/")
    dest = out / args.split
    manifest = load_manifest(ROOT, args.split, args.subset)
    jobs = [(r, methods, cfg, dest) for r in manifest.to_dict("records")]
    with ProcessPoolExecutor(args.jobs) as ex:
        records = [r for chunk in ex.map(_export_one, jobs, chunksize=4) for r in chunk]
    pd.DataFrame(records).to_csv(dest / f"{args.split}_outputs.csv", index=False,
                                 encoding="utf-8-sig")
    (dest / "config.json").write_text(json.dumps(
        {"config_file": str(args.config), "methods": methods, "params": cfg.to_dict()},
        indent=2), encoding="utf-8")
    print(f"{len(manifest)} files x {len(methods)} methods -> {dest}")


if __name__ == "__main__":
    main()
