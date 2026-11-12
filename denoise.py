"""Submission entry point: denoise every WAV in input/ and write it to output/.

    python denoise.py                          # ./input -> ./output (next to this file)
    python denoise.py --input IN --output OUT  # other folders
    python denoise.py --config configs/xxx.json

Each output keeps the input's file name (and sub-folder), sample rate, channel
count, sample count and sample format (e.g. PCM_16). Multi-channel files are
processed channel by channel. Algorithm parameters come from
configs/submission.json unless --config is given. If a file cannot be
processed, it is copied unchanged so that every input still has an output.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from nr import enhance, load_config  # noqa: E402

DEFAULT_CONFIG = HERE / "configs" / "submission.json"


def denoise_file(src: Path, dst: Path, method: str, cfg) -> None:
    info = sf.info(str(src))
    x, sr = sf.read(str(src), dtype="float64", always_2d=True)  # (samples, channels)
    y = np.empty_like(x)
    for ch in range(x.shape[1]):
        y[:, ch] = enhance(x[:, ch], sr, method, cfg)
    if not info.subtype.startswith(("FLOAT", "DOUBLE")):
        y = np.clip(y, -1.0, 1.0 - 2.0 ** -15)       # integer formats: no wrap-around
    if info.channels == 1:
        y = y[:, 0]
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), y, sr, subtype=info.subtype, format=info.format)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=HERE / "input")
    ap.add_argument("--output", type=Path, default=HERE / "output")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = ap.parse_args(argv)

    method, cfg = load_config(args.config)
    src_root = args.input.resolve()
    if not src_root.is_dir():
        print(f"Input folder not found: {src_root}", file=sys.stderr)
        return 1
    files = sorted(p for p in src_root.rglob("*")
                   if p.is_file() and p.suffix.lower() == ".wav")
    if not files:
        print(f"No WAV files in {src_root}", file=sys.stderr)
        return 1

    passed_through, failed = 0, 0
    for i, src in enumerate(files, 1):
        dst = args.output / src.relative_to(src_root)
        try:
            denoise_file(src, dst, method, cfg)
        except Exception as e:
            # Every input must get an output: fall back to the unprocessed file.
            print(f"[WARN] {src.name}: {type(e).__name__}: {e} -> copied unprocessed",
                  file=sys.stderr)
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)
                passed_through += 1
            except Exception as e2:
                failed += 1
                print(f"[FAIL] {src.name}: {type(e2).__name__}: {e2}", file=sys.stderr)
        if i % 50 == 0 or i == len(files):
            print(f"{i}/{len(files)} done", flush=True)
    print(f"method={method}  denoised={len(files) - passed_through - failed}  "
          f"unprocessed={passed_through}  failed={failed}  -> {args.output.resolve()}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
