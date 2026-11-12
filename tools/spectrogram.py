"""Spectrograms and listening files for the AI usage report's "분석 및 고찰".

    python tools/spectrogram.py speaker015_utt01_seg0001
    python tools/spectrogram.py --pick worst --run results/submissions/round1 --n 3
    python tools/spectrogram.py <id> --config configs/X.json --compare configs/submission.json

For each segment, writes to results/figures/<name>/ (git-ignored):
  <id>.png                 clean / noisy / enhanced (/ compared) spectrograms, PESQ·SNR in titles
  <id>_clean.wav, _noisy.wav, _<label>.wav   for listening (same level, 16-bit)

--pick worst|best|median takes segments from a run's per_file.csv by PESQ change,
so the figures show where the method helps and where it fails.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluate import pesq_wb, snr_db  # noqa: E402
from nr import enhance, load_config  # noqa: E402

SUBMISSION = ROOT / "configs" / "submission.json"


def spec_db(x, sr, n_fft=512, hop=128):
    from scipy import signal
    _, _, z = signal.stft(x, fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop)
    return 20 * np.log10(np.abs(z) + 1e-8)


def pick_segments(run: Path, how: str, n: int) -> list[str]:
    df = pd.read_csv(run / "per_file.csv", encoding="utf-8-sig")
    method = [m for m in df.method.unique() if m != "noisy_input"][-1]
    d = df[df.method == method].sort_values("delta_pesq")
    if how == "worst":
        d = d.head(n)
    elif how == "best":
        d = d.tail(n)
    else:
        mid = len(d) // 2
        d = d.iloc[max(0, mid - n // 2): mid - n // 2 + n]
    return d.segment_id.tolist()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("segments", nargs="*", help="segment_id(s) from samples.csv")
    ap.add_argument("--split", default="validation", choices=["train", "validation"])
    ap.add_argument("--config", type=Path, default=SUBMISSION)
    ap.add_argument("--compare", type=Path, help="second config to show (e.g. previous round)")
    ap.add_argument("--pick", choices=["worst", "best", "median"])
    ap.add_argument("--run", type=Path, help="run folder for --pick (default: latest round)")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--name", help="output folder name under results/figures/")
    ap.add_argument("--fmax", type=float, default=8000)
    args = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    manifest = pd.read_csv(ROOT / args.split / "samples.csv", encoding="utf-8-sig") \
        .set_index("segment_id")
    segments = list(args.segments)
    if args.pick:
        run = args.run or max((ROOT / "results" / "submissions").glob("round*"),
                              key=lambda p: int(p.name[5:]))
        segments += pick_segments(run, args.pick, args.n)
    if not segments:
        ap.error("give segment ids or --pick")

    configs = [(args.config.stem, *load_config(args.config))]
    if args.compare:
        configs.append((args.compare.stem, *load_config(args.compare)))
    out_dir = ROOT / "results" / "figures" / (args.name or "_vs_".join(c[0] for c in configs))
    out_dir.mkdir(parents=True, exist_ok=True)

    for seg in segments:
        row = manifest.loc[seg]
        clean, sr = sf.read(str(ROOT / row.clean_path), dtype="float64")
        noisy, _ = sf.read(str(ROOT / row.noisy_path), dtype="float64")
        panels = [("clean", clean, None), ("noisy", noisy, (pesq_wb(clean, noisy, sr),
                                                            snr_db(clean, noisy)))]
        for label, method, cfg in configs:
            y = enhance(noisy, sr, method, cfg)
            panels.append((f"{label} ({method})", y, (pesq_wb(clean, y, sr), snr_db(clean, y))))
            sf.write(str(out_dir / f"{seg}_{label}.wav"), np.clip(y, -1, 1 - 2 ** -15), sr,
                     subtype="PCM_16")
        sf.write(str(out_dir / f"{seg}_clean.wav"), clean, sr, subtype="PCM_16")
        sf.write(str(out_dir / f"{seg}_noisy.wav"), noisy, sr, subtype="PCM_16")

        fig, axes = plt.subplots(len(panels), 1, figsize=(10, 2.3 * len(panels)),
                                 sharex=True, constrained_layout=True)
        vmax = spec_db(clean, sr).max()
        for ax, (title, x, score) in zip(axes, panels):
            s = spec_db(x, sr)
            im = ax.imshow(s, origin="lower", aspect="auto", cmap="magma",
                           vmin=vmax - 80, vmax=vmax,
                           extent=[0, len(x) / sr, 0, sr / 2 / 1000])
            if score:
                title += f"   PESQ {score[0]:.2f} · SNR {score[1]:.1f} dB"
            ax.set_title(title, fontsize=10, loc="left")
            ax.set_ylabel("kHz")
            ax.set_ylim(0, args.fmax / 1000)
        axes[-1].set_xlabel("time (s)")
        fig.colorbar(im, ax=axes, label="dB", shrink=0.8)
        fig.suptitle(f"{seg} · {row.noise_category} ({row.noise_code}, {row.noise_difficulty}) · "
                     f"target {row.target_snr_db:g} dB", fontsize=11)
        fig.savefig(out_dir / f"{seg}.png", dpi=110)
        plt.close(fig)
        print(f"{seg}: {row.noise_category} {row.target_snr_db:g} dB -> {out_dir / (seg + '.png')}")
    (out_dir / "configs.json").write_text(json.dumps(
        {c[0]: {"method": c[1], "params": c[2].to_dict()} for c in configs}, indent=2),
        encoding="utf-8")


if __name__ == "__main__":
    main()
