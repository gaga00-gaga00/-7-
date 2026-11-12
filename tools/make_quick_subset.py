"""Create splits/train_quick.txt: a fixed, stratified ~200-file subset of train.

Strata = noise_difficulty x target_snr_db, sampled in proportion with a fixed
seed, so tuning runs are fast and always use the same files.

    python tools/make_quick_subset.py [--n 200] [--seed 0]
    python tools/make_quick_subset.py --name s400 --n 400 --by noise_category target_snr_db --exact
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="quick", help="writes splits/train_<name>.txt")
    ap.add_argument("--by", nargs="+", default=["noise_difficulty", "target_snr_db"],
                    help="manifest columns that define the strata")
    ap.add_argument("--exact", action="store_true",
                    help="largest-remainder allocation: exactly --n files")
    args = ap.parse_args(argv)

    m = pd.read_csv(ROOT / "train" / "samples.csv", encoding="utf-8-sig")
    rng = np.random.default_rng(args.seed)
    picked = []
    groups = [g for _, g in m.groupby(args.by)]
    quota = np.array([args.n * len(g) / len(m) for g in groups])
    if args.exact:
        counts = np.floor(quota).astype(int)
        for i in np.argsort(-(quota - counts), kind="stable")[:args.n - counts.sum()]:
            counts[i] += 1
    else:
        counts = np.maximum(1, np.round(quota)).astype(int)
    for g, k in zip(groups, counts):
        picked += list(rng.choice(g.segment_id.sort_values().to_numpy(), size=min(k, len(g)),
                                  replace=False))
    picked = sorted(picked)
    out = ROOT / "splits" / f"train_{args.name}.txt"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(picked) + "\n", encoding="utf-8")
    sub = m[m.segment_id.isin(picked)]
    print(f"{len(picked)} files -> {out}")
    print(sub.groupby(args.by).size().to_string())


if __name__ == "__main__":
    main()
