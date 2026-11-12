"""Create splits/train_quick.txt: a fixed, stratified ~200-file subset of train.

Strata = noise_difficulty x target_snr_db, sampled in proportion with a fixed
seed, so tuning runs are fast and always use the same files.

    python tools/make_quick_subset.py [--n 200] [--seed 0]
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
    args = ap.parse_args(argv)

    m = pd.read_csv(ROOT / "train" / "samples.csv", encoding="utf-8-sig")
    rng = np.random.default_rng(args.seed)
    picked = []
    for _, g in m.groupby(["noise_difficulty", "target_snr_db"]):
        k = max(1, round(args.n * len(g) / len(m)))
        picked += list(rng.choice(g.segment_id.sort_values().to_numpy(), size=min(k, len(g)),
                                  replace=False))
    picked = sorted(picked)
    out = ROOT / "splits" / "train_quick.txt"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(picked) + "\n", encoding="utf-8")
    sub = m[m.segment_id.isin(picked)]
    print(f"{len(picked)} files -> {out}")
    print(sub.groupby(["noise_difficulty", "target_snr_db"]).size().to_string())


if __name__ == "__main__":
    main()
