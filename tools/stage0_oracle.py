"""Stage 0: is per-file adaptation worth it?

Four fixed settings with different characters are applied to the same train
subset. If picking the best setting per file (oracle, needs clean speech) is
not clearly better than the best single setting, per-file noise classification
cannot pay off.

    S1 base     MMSE-LSA, per-file noise spectrum (mean of the lowest-energy frames)
    S2 strong   noise over-estimation x2, gain floor 0.02 (-34 dB)
    S3 weak     no over-estimation, gain floor 0.25 (-12 dB)
    S4 lowcut   S1 followed by a 300 Hz high-pass

Every output is written as a 16-bit WAV and read back before scoring.
Clean speech and noise labels are used for scoring/reporting only.

--base spp puts the same four characters on the SPP candidate (configs/omlsa.json)
and adds noise-estimation variants N1-N5.

    python tools/stage0_oracle.py [--base spp|lsa] [--subset s400]   # default: all of train
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
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluate import load_manifest, pesq_score, read_checked  # noqa: E402
from nr import DEFAULT, enhance  # noqa: E402
from nr.methods import butterworth_stages  # noqa: E402

LSA = dict(win_sec=0.032, hop_sec=0.008, noise_fraction=0.2, dd_alpha=0.98,
           lsa_xi_min_db=-25.0, lsa_gain_floor=0.05, noise_scale=1.0)
SPP = json.loads((ROOT / "configs" / "omlsa.json").read_text(encoding="utf-8"))["params"]
# base -> (method, base params, {name: (overrides, post high-pass Hz or None)})
# The first four settings are the "characters" of the stage-0 plan; N* (spp base
# only) change how the noise PSD is estimated instead of how hard it is removed.
BASES = {
    "lsa": ("mmse_lsa", LSA, {
        "S1_base": ({}, None),
        "S2_strong": (dict(noise_scale=2.0, lsa_gain_floor=0.02), None),
        "S3_weak": (dict(lsa_gain_floor=0.25), None),
        "S4_lowcut": ({}, 300.0),
    }),
    "spp": ("omlsa", SPP, {
        "S1_base": ({}, None),
        # with omlsa_q = 0.01 the OM-LSA floor is almost never applied, so the
        # depth of suppression is set by the a-priori SNR floor instead:
        # xi_min -10 dB -> LSA gain about -12 dB, -35 dB -> about -37 dB
        "S2_strong": (dict(noise_scale=2.0, lsa_xi_min_db=-35.0), None),
        "S3_weak": (dict(lsa_xi_min_db=-10.0), None),
        "S4_lowcut": ({}, 300.0),
        "N1_fast": (dict(spp_tau_sec=0.1), None),
        "N2_slow": (dict(spp_tau_sec=1.0), None),
        "N3_fixed": (dict(noise_estimator="lowest_energy"), None),
        "N4_fsmooth": (dict(noise_smooth_bins=5), None),
        "N5_quantile": (dict(noise_estimator="quantile", quant_window_sec=1.5), None),
    }),
}
CHARS = ["S1_base", "S2_strong", "S3_weak", "S4_lowcut"]


def process(noisy, sr, base, name):
    method, params, settings = BASES[base]
    over, hp = settings[name]
    cfg = DEFAULT.with_overrides(**{**params, **over})
    y = enhance(noisy, sr, method, cfg)
    if hp:
        y = butterworth_stages(y, sr, cfg.with_overrides(highpass_hz=hp), ("highpass",))
    return y


def roundtrip_16bit(y, sr, path):
    """Write a 16-bit PCM WAV and read it back (what the grader will see)."""
    sf.write(str(path), y, sr, subtype="PCM_16", format="WAV")
    out, _ = sf.read(str(path), dtype="float64")
    return out


def _job(args):
    root, row, base = args
    clean, sr = read_checked(root, row, "clean")
    noisy, _ = read_checked(root, row, "noisy")
    rec = {k: row[k] for k in ("segment_id", "noise_category", "noise_code",
                               "noise_difficulty", "target_snr_db")}
    rec["noisy"] = pesq_score(clean, noisy, sr)[0]
    with tempfile.TemporaryDirectory() as tmp:
        for name in BASES[base][2]:
            y = roundtrip_16bit(process(noisy, sr, base, name), sr, Path(tmp) / f"{name}.wav")
            rec[name] = pesq_score(clean, y, sr)[0]
    return rec


def se(x):
    return x.std(ddof=1) / np.sqrt(len(x))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="spp", choices=list(BASES))
    ap.add_argument("--split", default="train")
    ap.add_argument("--subset", help="splits/<split>_<subset>.txt (default: every file)")
    ap.add_argument("--out", type=Path, help="default: results/stage0_<base>_<subset|all>")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args(argv)
    names = list(BASES[args.base][2])
    out = args.out or ROOT / "results" / f"stage0_{args.base}_{args.subset or 'all'}"

    manifest = load_manifest(ROOT, args.split, args.subset)
    jobs = [(ROOT, r, args.base) for r in manifest.to_dict("records")]
    with ProcessPoolExecutor(args.jobs) as ex:
        df = pd.DataFrame(list(ex.map(_job, jobs, chunksize=4)))
    if df[names].isna().any().any():
        raise RuntimeError("PESQ failed on some files; see the NaN rows.")

    df["oracle4"] = df[CHARS].max(axis=1)
    df["oracle"] = df[names].max(axis=1)
    df["oracle_pick"] = df[names].idxmax(axis=1)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "per_file.csv", index=False, encoding="utf-8-sig")

    cols = ["noisy"] + names + ["oracle4", "oracle"]
    means = df[cols].mean()
    best4, best = means[CHARS].idxmax(), means[names].idxmax()
    g4, g = df.oracle4 - df[best4], df.oracle - df[best]
    lines = [f"{args.split}/{args.subset or 'all'}: {len(df)} files, base {args.base}, "
             "16-bit WAV round trip", "", means.round(4).to_string(), "",
             f"S1-S4  best single {best4} {means[best4]:.4f} | oracle {means.oracle4:.4f} | "
             f"gap {g4.mean():.4f} (SE {se(g4):.4f})"]
    if len(names) > len(CHARS):
        lines.append(f"all {len(names)}  best single {best} {means[best]:.4f} | "
                     f"oracle {means.oracle:.4f} | gap {g.mean():.4f} (SE {se(g):.4f})")
    lines += ["", "paired difference vs S1_base (mean, SE, oracle picks, "
                  "oracle of {S1, this})"]
    picks = df.oracle_pick.value_counts()
    for n in names:
        d = df[n] - df.S1_base
        lines.append(f"  {n:12s} {d.mean():+.4f}  {se(d):.4f}  {int(picks.get(n, 0)):5d}  "
                     f"{df[['S1_base', n]].max(axis=1).mean():.4f}")
    for grp in ("target_snr_db", "noise_category"):
        tab = df.groupby(grp)[cols].mean()
        tab.insert(0, "n", df.groupby(grp).size())
        tab["gap"] = tab.oracle - tab[best]
        tab.to_csv(out / f"by_{grp}.csv", encoding="utf-8-sig")
        lines += ["", f"by {grp}", tab.round(3).to_string()]
    text = "\n".join(lines)
    (out / "summary.txt").write_text(text + "\n", encoding="utf-8")
    pd.set_option("display.width", 250)
    print(text)


if __name__ == "__main__":
    main()
