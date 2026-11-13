"""Stage 2: does each category-specific logic help the files of its category?

Every variant is applied at full strength to every train file, scored on a
16-bit WAV round trip, and compared file by file with the base (C0 =
configs/omlsa.json). The in-category change is then read off for several
thresholds of the feature that defines the category (nr/features.py, noisy
input only). Clean speech and noise labels are used for scoring/reporting only.

    python tools/stage2_modules.py [--subset s400]

Adoption rule: paired mean difference d >= 0.003 and d >= 2 * standard error.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluate import load_manifest, pesq_score, read_checked  # noqa: E402
from nr import DEFAULT, enhance  # noqa: E402
from nr.features import features  # noqa: E402
from tools.stage0_oracle import roundtrip_16bit  # noqa: E402

BASE = json.loads((ROOT / "configs" / "omlsa.json").read_text(encoding="utf-8"))["params"]
VARIANTS = {
    "base": {},
    "tone": dict(tone_weight=1.0),                                   # C1
    "impulse": dict(impulse_weight=1.0),                             # C2
    "tone_w02": dict(tone_weight=0.2),
    "impulse_w02": dict(impulse_weight=0.2),
    "hp100": dict(post_highpass_hz=100.0),                           # C3
    "hp150": dict(post_highpass_hz=150.0),
    "hp200": dict(post_highpass_hz=200.0),
    "slow_sc08": dict(spp_tau_sec=1.0, noise_scale=0.8),             # C4 as planned
    "fast_sc08": dict(spp_tau_sec=0.1, noise_scale=0.8),             # C4 the other way
}
# stage-0 settings that are reused as variants (same base, same scoring)
STAGE0 = {"S4_lowcut": "hp300", "N1_fast": "fast", "N2_slow": "slow", "N3_fixed": "fixed",
          "S2_strong": "strong"}
# (variant, feature, side, thresholds): the files with feature >= / <= threshold
TESTS = [
    ("tone", "tone_persist", ">=", (0.0, 0.25, 0.5, 0.75, 0.9)),
    ("impulse", "impulsive", ">=", (0.0, 0.005, 0.02, 0.04)),
    ("tone_w02", "tone_persist", ">=", (0.0, 0.25, 0.5, 0.75, 0.9)),
    ("impulse_w02", "impulsive", ">=", (0.0, 0.005, 0.02, 0.04)),
    ("hp100", "low_ratio", ">=", (0.0, 0.4, 0.6, 0.8, 0.9)),
    ("hp150", "low_ratio", ">=", (0.0, 0.4, 0.6, 0.8, 0.9)),
    ("hp200", "low_ratio", ">=", (0.0, 0.4, 0.6, 0.8, 0.9)),
    ("hp300", "low_ratio", ">=", (0.0, 0.4, 0.6, 0.8, 0.9)),
    ("slow_sc08", "variability", ">=", (0.0, 1.0, 1.8, 3.0)),
    ("slow", "variability", ">=", (0.0, 1.0, 1.8, 3.0)),
    ("fast_sc08", "variability", ">=", (0.0, 1.0, 1.8, 3.0)),
    ("fast", "variability", ">=", (0.0, 1.0, 1.8, 3.0)),
    ("fast", "flatness", "<=", (1.0, 0.25, 0.12, 0.05)),
    ("fixed", "flatness", ">=", (0.0, 0.25, 0.4, 0.6)),
    ("slow", "flatness", ">=", (0.0, 0.25, 0.4, 0.6)),
    ("strong", "low_ratio", "<=", (1.0, 0.4, 0.17, 0.07)),
    ("strong", "tone_persist", ">=", (0.0, 0.5, 0.9)),
]


def _job(args):
    root, row, names = args
    clean, sr = read_checked(root, row, "clean")
    noisy, _ = read_checked(root, row, "noisy")
    rec = {k: row[k] for k in ("segment_id", "noise_category", "noise_difficulty",
                               "target_snr_db")}
    rec.update(features(noisy, sr))
    with tempfile.TemporaryDirectory() as tmp:
        for name in names:
            over = VARIANTS[name]
            cfg = DEFAULT.with_overrides(**{**BASE, **over})
            y = roundtrip_16bit(enhance(noisy, sr, "omlsa", cfg), sr, Path(tmp) / f"{name}.wav")
            rec[name] = pesq_score(clean, y, sr)[0]
    return rec


def report(df, names):
    n_all = len(df)
    lines = ["mean PESQ", df[["base"] + names].mean().round(4).to_string(), "",
             "in-category paired change vs base  (d, SE, d/SE, share of files, "
             "effect on the overall mean)"]
    for variant, feat, side, thresholds in TESTS:
        if variant not in df:
            continue
        lines.append(f"\n{variant}  on  {feat} {side} threshold")
        d_all = df[variant] - df.base
        for thr in thresholds:
            sel = df[feat] >= thr if side == ">=" else df[feat] <= thr
            d = d_all[sel]
            if len(d) < 2:
                continue
            se = d.std(ddof=1) / np.sqrt(len(d))
            ok = "ADOPT" if d.mean() >= 0.003 and d.mean() >= 2 * se else ""
            lines.append(f"  {thr:6.3f}  n={len(d):5d}  d={d.mean():+.4f}  SE={se:.4f}  "
                         f"t={d.mean() / se if se > 0 else 0:+5.1f}  "
                         f"overall {d.sum() / n_all:+.4f}  {ok}")
    cat = df.groupby("noise_category")[["base"] + names].mean()
    cat = cat.sub(cat.base, axis=0).drop(columns="base")
    cat.insert(0, "base", df.groupby("noise_category").base.mean())
    cat.insert(0, "n", df.groupby("noise_category").size())
    lines += ["", "change vs base by noise label (applied to every file)", cat.round(3).to_string()]
    return lines, cat


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="train")
    ap.add_argument("--subset", help="splits/<split>_<subset>.txt (default: every file)")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--report-only", action="store_true", help="re-read per_file.csv")
    ap.add_argument("--variants", nargs="+", choices=list(VARIANTS),
                    help="run only these and merge them into the existing per_file.csv")
    args = ap.parse_args(argv)
    out = ROOT / "results" / f"stage2_{args.split}_{args.subset or 'all'}"

    if args.report_only:
        df = pd.read_csv(out / "per_file.csv", encoding="utf-8-sig")
    else:
        manifest = load_manifest(ROOT, args.split, args.subset)
        run = args.variants or list(VARIANTS)
        with ProcessPoolExecutor(args.jobs) as ex:
            df = pd.DataFrame(list(ex.map(
                _job, [(ROOT, r, run) for r in manifest.to_dict("records")], chunksize=4)))
        if args.variants:                                  # keep the earlier columns
            old = pd.read_csv(out / "per_file.csv", encoding="utf-8-sig")
            new = df.set_index("segment_id")
            for name in run:
                old[name] = old.segment_id.map(new[name])
            df = old
        stage0 = ROOT / "results" / "stage0_spp_all" / "per_file.csv"
        if args.split == "train" and stage0.exists():
            s0 = pd.read_csv(stage0, encoding="utf-8-sig").set_index("segment_id")
            for src, name in STAGE0.items():
                df[name] = df.segment_id.map(s0[src])
            df["stage0_base"] = df.segment_id.map(s0["S1_base"])
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / "per_file.csv", index=False, encoding="utf-8-sig")

    names = [n for n in list(VARIANTS)[1:] + list(STAGE0.values()) if n in df]
    pd.set_option("display.width", 250)
    lines, cat = report(df, names)
    if "stage0_base" in df:
        lines.insert(0, f"check: max |base - stage-0 base| = "
                        f"{(df.base - df.stage0_base).abs().max():.4f}\n")
    lines.insert(0, f"{args.split}/{args.subset or 'all'}: {len(df)} files, 16-bit WAV round trip")
    cat.to_csv(out / "by_category.csv", encoding="utf-8-sig")
    text = "\n".join(lines)
    (out / "summary.txt").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
