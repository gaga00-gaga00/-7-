"""Where is the PESQ lost? Oracle ladder on a train subset (analysis only).

Each rung replaces one estimated quantity of the current candidate
(configs/omlsa.json) with its true value, computed from clean speech. None of
this can be used for inference; it only tells which part limits the score.

    cand        current candidate (estimated noise PSD, decision-directed gain)
    noise_mean  same gain rule, true noise PSD averaged over the file (stationary)
    noise_2s / noise_1s / noise_05s   true noise PSD smoothed over 2 / 1 / 0.5 s
    noise_tv    same gain rule, true noise PSD following time (smoothed 0.1 s)
    noise_absent  noise tracked from the noisy power with perfect knowledge of
                where speech is absent (bound for SPP/MCRA-type blind trackers)
    irm         ideal Wiener gain sqrt(S^2/(S^2+N^2)) on the noisy STFT (best any
                real gain in [0,1] per bin can roughly do; noisy phase)
    irm_floor   irm limited to the candidate's gain floor
    clean_mag   clean magnitude + noisy phase (phase limit of the framework)

    python tools/diagnose_gap.py [--subset s400] [--out results/diagnose_all]   # default: all of train
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluate import load_manifest, pesq_score, read_checked, snr_db  # noqa: E402
from nr import load_config  # noqa: E402
from nr.methods import omlsa, omlsa_gain_stft  # noqa: E402
from nr.noise import _alpha, _smooth_time  # noqa: E402
from nr.stft import frame_params, istft, stft  # noqa: E402
from tools.stage0_oracle import roundtrip_16bit  # noqa: E402

RUNGS = ["cand", "noise_mean", "noise_2s", "noise_1s", "noise_05s", "noise_tv",
         "noise_absent", "irm", "irm_floor", "clean_mag"]
EPS = 1e-12


def absent_tracker(py, ps, pn, a):
    """Noise PSD of a tracker with perfect speech-presence knowledge.

    It may look at the noisy power only in cells where speech is truly absent
    (clean power 10 dB below the noise) and has to fill the rest from the
    neighbouring absent cells in time. This is what SPP / MCRA approximate
    blindly, so it bounds that whole family of trackers.
    """
    absent = (ps < 0.1 * pn).astype(float)
    num = _smooth_time(py * absent, a, zero_phase=True)
    den = _smooth_time(absent, a, zero_phase=True)
    est = num / np.maximum(den, 1e-6)
    # bins with (almost) no absent cell nearby: fall back to the file-wide value
    glob = (py * absent).sum(axis=1, keepdims=True) / np.maximum(
        absent.sum(axis=1, keepdims=True), 1.0)
    glob = np.where(absent.sum(axis=1, keepdims=True) > 0, glob, py.mean(axis=1, keepdims=True))
    w = np.minimum(den / 0.05, 1.0)
    return np.maximum(w * est + (1 - w) * glob, EPS)


def _job(args):
    root, row, cfg = args
    clean, sr = read_checked(root, row, "clean")
    noisy, _ = read_checked(root, row, "noisy")
    n = len(noisy)
    _, zy = stft(noisy, sr, cfg)
    _, zs = stft(clean, sr, cfg)
    _, zn = stft(noisy - clean, sr, cfg)
    py, ps, pn = np.abs(zy) ** 2, np.abs(zs) ** 2, np.abs(zn) ** 2
    hop_sec = frame_params(sr, cfg)[1] / sr
    irm = np.sqrt(ps / (ps + pn + EPS))

    def with_noise(psd):
        """The candidate's gain rule on a noise PSD supplied by this script.

        The true noise is injected here only; nr/ never sees clean speech.
        """
        return istft(omlsa_gain_stft(zy, np.maximum(psd, EPS), cfg), sr, n, cfg)

    def true_noise(tau):
        return _smooth_time(pn, _alpha(tau, hop_sec), zero_phase=True)

    out = {
        "cand": omlsa(noisy, sr, cfg),
        "noise_mean": with_noise(pn.mean(axis=1, keepdims=True) * np.ones_like(pn)),
        "noise_2s": with_noise(true_noise(2.0)),
        "noise_1s": with_noise(true_noise(1.0)),
        "noise_05s": with_noise(true_noise(0.5)),
        "noise_tv": with_noise(true_noise(0.1)),
        "noise_absent": with_noise(absent_tracker(py, ps, pn, _alpha(0.3, hop_sec))),
        "irm": istft(irm * zy, sr, n, cfg),
        "irm_floor": istft(np.maximum(irm, cfg.omlsa_gain_floor) * zy, sr, n, cfg),
        "clean_mag": istft(np.abs(zs) * np.exp(1j * np.angle(zy)), sr, n, cfg),
    }
    rec = {k: row[k] for k in ("segment_id", "noise_category", "noise_difficulty",
                               "target_snr_db")}
    rec["input_snr_db"] = snr_db(clean, noisy)
    rec["noisy"] = pesq_score(clean, noisy, sr)[0]
    # how much the noise level moves inside the file (dB std of frame noise power)
    frame = 10 * np.log10(pn.mean(axis=0) + EPS)
    rec["noise_level_std_db"] = float(np.std(frame[frame > frame.max() - 60]))
    with tempfile.TemporaryDirectory() as tmp:
        for name in RUNGS:
            y = roundtrip_16bit(out[name], sr, Path(tmp) / f"{name}.wav")
            rec[name] = pesq_score(clean, y, sr)[0]
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="train")
    ap.add_argument("--subset", help="splits/<split>_<subset>.txt (default: every file)")
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "omlsa.json")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "diagnose_all")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args(argv)

    _, cfg = load_config(args.config)
    manifest = load_manifest(ROOT, args.split, args.subset)
    with ProcessPoolExecutor(args.jobs) as ex:
        df = pd.DataFrame(list(ex.map(_job, [(ROOT, r, cfg) for r in manifest.to_dict("records")],
                                      chunksize=4)))
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "per_file.csv", index=False, encoding="utf-8-sig")

    cols = ["noisy"] + RUNGS
    pd.set_option("display.width", 220)
    lines = [f"{args.split}/{args.subset or 'all'}: {len(df)} files, config {args.config.name}", "",
             df[cols].mean().round(4).to_string()]
    df["var_bin"] = pd.cut(df.noise_level_std_db, [0, 2, 4, 8, 100],
                           labels=["<2dB", "2-4dB", "4-8dB", ">8dB"])
    for g in ("target_snr_db", "var_bin", "noise_category"):
        tab = df.groupby(g, observed=True)[cols].mean()
        tab.insert(0, "n", df.groupby(g, observed=True).size())
        tab.to_csv(args.out / f"by_{g}.csv", encoding="utf-8-sig")
        lines += ["", f"by {g}", tab.round(3).to_string()]
    text = "\n".join(lines)
    (args.out / "summary.txt").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
