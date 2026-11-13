"""Stage 1: noise-character features (nr/features.py) per file and per noise label.

The features are computed from the noisy waveform only. The noise labels and
the stage-0 scores are used here for analysis, never by the inference code.

    python tools/stage1_features.py [--split train] [--subset s400]

Writes results/stage1_<split>_<subset|all>/features.csv, by_category.csv and
summary.txt. If results/stage0_spp_all/per_file.csv exists (train), it also
reports how well each feature predicts which stage-0 setting beats the base.
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluate import load_manifest, read_checked  # noqa: E402
from nr.features import NAMES, features  # noqa: E402

F = list(NAMES)
LABEL = {"flatness": "F1", "tone_persist": "F2", "tone_drift": "F3", "impulsive": "F4",
         "variability": "F5", "low_ratio": "F6"}


def _job(args):
    root, row = args
    noisy, sr = read_checked(root, row, "noisy")
    rec = {k: row[k] for k in ("segment_id", "noise_category", "noise_difficulty",
                               "target_snr_db")}
    rec.update(features(noisy, sr))
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="train")
    ap.add_argument("--subset", help="splits/<split>_<subset>.txt (default: every file)")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args(argv)
    out = ROOT / "results" / f"stage1_{args.split}_{args.subset or 'all'}"

    manifest = load_manifest(ROOT, args.split, args.subset)
    with ProcessPoolExecutor(args.jobs) as ex:
        df = pd.DataFrame(list(ex.map(_job, [(ROOT, r) for r in manifest.to_dict("records")],
                                      chunksize=8)))
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "features.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 250)
    head = [f"{LABEL[c]} {c}" for c in F]
    lines = [f"{args.split}/{args.subset or 'all'}: {len(df)} files", "",
             "all files (percentiles)",
             df[F].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).set_axis(head, axis=1).round(3).to_string()]
    med = df.groupby("noise_category")[F].median()
    med.insert(0, "n", df.groupby("noise_category").size())
    med.to_csv(out / "by_category.csv", encoding="utf-8-sig")
    lines += ["", "median by noise label", med.round(3).to_string()]
    for q, name in ((0.25, "25th percentile"), (0.75, "75th percentile")):
        tab = df.groupby("noise_category")[F].quantile(q)
        tab.to_csv(out / f"by_category_q{int(q * 100)}.csv", encoding="utf-8-sig")
        lines += ["", f"{name} by noise label", tab.round(3).to_string()]
    snr = df.groupby("target_snr_db")[F].median()
    lines += ["", "median by SNR (a noise feature should not depend on it much)",
              snr.round(3).to_string()]

    stage0 = ROOT / "results" / "stage0_spp_all" / "per_file.csv"
    if args.split == "train" and stage0.exists():
        s0 = pd.read_csv(stage0, encoding="utf-8-sig").set_index("segment_id")
        m = df.set_index("segment_id").join(s0.drop(columns=[c for c in s0 if c in df.columns]),
                                            how="inner")
        gains = {n: m[n] - m.S1_base for n in ("S2_strong", "S3_weak", "S4_lowcut", "N1_fast",
                                               "N2_slow", "N3_fixed") if n in m}
        corr = pd.DataFrame({n: [m[c].corr(g, method="spearman") for c in F]
                             for n, g in gains.items()}, index=head)
        corr.to_csv(out / "feature_vs_stage0_gain.csv", encoding="utf-8-sig")
        lines += ["", f"Spearman correlation of each feature with (setting - S1_base), n={len(m)}",
                  corr.round(2).to_string()]
    text = "\n".join(lines)
    (out / "summary.txt").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
