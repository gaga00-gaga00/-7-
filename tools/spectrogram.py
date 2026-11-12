"""Spectrograms and listening files (course notebook's show_example / inspect_sample).

    python tools/spectrogram.py speaker015_utt01_seg0001
    python tools/spectrogram.py --pick worst --run results/submissions/round1 --n 3
    python tools/spectrogram.py <id> --config configs/X.json --compare configs/submission.json
    python tools/spectrogram.py <id> --methods basic_filter spectral_subtraction wiener_filter mmse_lsa
    python tools/spectrogram.py --noisy some/file.wav [--clean ref.wav]   # any file
    python tools/spectrogram.py <id> --added-noise --no-enhance            # data exploration

For each input, writes to results/figures/<name>/ (git-ignored):
  <id>.png       clean / noisy / [added noise] / enhanced panels, PESQ·SNR when clean exists
  <id>_*.wav     listening files, all scaled by ONE common gain so levels stay comparable

Every panel uses the same dB reference (the noisy STFT peak), like the notebook.
--pick worst|best|median takes segments from a run's per_file.csv by PESQ change.
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
from nr import DEFAULT, enhance, load_config  # noqa: E402

SUBMISSION = ROOT / "configs" / "submission.json"


def stft_mag(x, sr, n_fft=512, hop=128):
    from scipy import signal
    f, t, z = signal.stft(x, fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop,
                          nfft=n_fft, boundary="zeros", padded=True)
    return f, t, np.abs(z)


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


def panel_label(label: str, method: str) -> str:
    return method if label == method else f"{label} ({method})"


def read_mono(path: Path):
    x, sr = sf.read(str(path), dtype="float64", always_2d=True)
    return x.mean(axis=1), sr


def render(name: str, title: str, noisy, sr, clean, outputs: list[tuple[str, np.ndarray]],
           out_dir: Path, added_noise: bool, fmax: float):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = []
    if clean is not None:
        panels.append(("clean", clean, None))
    panels.append(("noisy", noisy, (pesq_wb(clean, noisy, sr), snr_db(clean, noisy))
                   if clean is not None else None))
    if added_noise and clean is not None:
        panels.append(("added noise (noisy - clean)", noisy - clean, None))
    for label, y in outputs:
        panels.append((label, y, (pesq_wb(clean, y, sr), snr_db(clean, y))
                       if clean is not None else None))

    ref = max(float(stft_mag(noisy, sr)[2].max()), 1e-12)
    fig, axes = plt.subplots(len(panels), 1, figsize=(10, 2.2 * len(panels)), sharex=True,
                             sharey=True, constrained_layout=True, squeeze=False)
    axes = axes[:, 0]
    for ax, (label, x, score) in zip(axes, panels):
        f, t, mag = stft_mag(x, sr)
        db = 20 * np.log10(np.maximum(mag, 1e-12) / ref)
        mesh = ax.pcolormesh(t, f / 1000, db, shading="auto", cmap="magma", vmin=-80, vmax=0)
        if score:
            label += f"   PESQ {score[0]:.2f} · SNR {score[1]:.1f} dB"
        ax.set_title(label, fontsize=10, loc="left")
        ax.set_ylabel("kHz")
        ax.set_ylim(0, min(fmax, sr / 2) / 1000)
    axes[-1].set_xlabel("time (s)")
    fig.colorbar(mesh, ax=list(axes), label="dB re noisy STFT peak", shrink=0.8)
    fig.suptitle(title, fontsize=11)
    fig.savefig(out_dir / f"{name}.png", dpi=110)
    plt.close(fig)

    # one playback gain for every file keeps level differences audible
    gain = 0.99 / max(1.0, max(float(np.abs(x).max()) for _, x, _ in panels))
    for label, x, _ in panels:
        tag = label.split(" (")[0].split(" ")[0].replace("/", "_")
        sf.write(str(out_dir / f"{name}_{tag}.wav"), x * gain, sr, subtype="PCM_16")
    return out_dir / f"{name}.png"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("segments", nargs="*", help="segment_id(s) from samples.csv")
    ap.add_argument("--split", default="validation", choices=["train", "validation"])
    ap.add_argument("--noisy", type=Path, help="any noisy WAV instead of segment ids")
    ap.add_argument("--clean", type=Path, help="optional clean reference for --noisy")
    ap.add_argument("--config", type=Path, default=SUBMISSION)
    ap.add_argument("--compare", type=Path, help="second config to show (e.g. previous round)")
    ap.add_argument("--methods", nargs="+",
                    help="show these methods (parameters from --config) instead of the "
                         "config's own method")
    ap.add_argument("--no-enhance", action="store_true", help="only clean/noisy(/added noise)")
    ap.add_argument("--added-noise", action="store_true", help="add a noisy - clean panel")
    ap.add_argument("--pick", choices=["worst", "best", "median"])
    ap.add_argument("--run", type=Path, help="run folder for --pick (default: latest round)")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--name", help="output folder name under results/figures/")
    ap.add_argument("--fmax", type=float, default=8000)
    args = ap.parse_args(argv)

    method, cfg = load_config(args.config)
    runs = []  # (label, method, cfg)
    if not args.no_enhance:
        if args.methods:
            runs = [(m, m, cfg) for m in args.methods]
        else:
            runs = [(args.config.stem, method, cfg)]
        if args.compare:
            m2, c2 = load_config(args.compare)
            runs.append((args.compare.stem + ("_b" if args.compare.stem == args.config.stem
                                              else ""), m2, c2))
    default_name = "explore" if args.no_enhance else "_vs_".join(r[0] for r in runs)
    out_dir = ROOT / "results" / "figures" / (args.name or default_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.noisy:
        noisy, sr = read_mono(args.noisy)
        clean = None
        if args.clean:
            clean, sr_c = read_mono(args.clean)
            if sr_c != sr or clean.shape != noisy.shape:
                ap.error("clean and noisy must have the same sample rate and length")
        outputs = [(panel_label(label, m), enhance(noisy, sr, m, c)) for label, m, c in runs]
        png = render(args.noisy.stem, args.noisy.name, noisy, sr, clean, outputs, out_dir,
                     args.added_noise, args.fmax)
        print(f"{args.noisy.name} -> {png}")
    else:
        manifest = pd.read_csv(ROOT / args.split / "samples.csv", encoding="utf-8-sig") \
            .set_index("segment_id")
        segments = list(args.segments)
        if args.pick:
            run = args.run or max((ROOT / "results" / "submissions").glob("round*"),
                                  key=lambda p: int(p.name[5:]))
            segments += pick_segments(run, args.pick, args.n)
        if not segments:
            ap.error("give segment ids, --pick or --noisy")
        for seg in segments:
            row = manifest.loc[seg]
            clean, sr = sf.read(str(ROOT / row.clean_path), dtype="float64")
            noisy, _ = sf.read(str(ROOT / row.noisy_path), dtype="float64")
            outputs = [(panel_label(label, m), enhance(noisy, sr, m, c)) for label, m, c in runs]
            title = (f"{seg} · {row.noise_category} ({row.noise_code}, {row.noise_difficulty}) · "
                     f"target {row.target_snr_db:g} dB")
            png = render(seg, title, noisy, sr, clean, outputs, out_dir, args.added_noise,
                         args.fmax)
            print(f"{seg}: {row.noise_category} {row.target_snr_db:g} dB -> {png}")
    (out_dir / "configs.json").write_text(json.dumps(
        {label: {"method": m, "params": c.to_dict()} for label, m, c in runs}
        or {"defaults": DEFAULT.to_dict()}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
