"""Submission contract tests. Uses synthetic audio only (no dataset needed).

    python -m pytest -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import denoise  # noqa: E402
from nr import DEFAULT, METHODS, enhance  # noqa: E402


def noisy_speechlike(n, sr, seed=0, amp=0.3):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / sr
    env = 0.5 * (1 + np.sin(2 * np.pi * 3 * t))            # syllable-like envelope
    voiced = env * sum(np.sin(2 * np.pi * f * t) / (i + 1)
                       for i, f in enumerate([150, 300, 450, 900, 1800]))
    return amp * voiced / np.max(np.abs(voiced) + 1e-9) + 0.05 * rng.standard_normal(n)


def run(tmp_path, files: dict[str, tuple[np.ndarray, int, str]]):
    inp, out = tmp_path / "input", tmp_path / "output"
    for rel, (x, sr, subtype) in files.items():
        p = inp / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(p), x, sr, subtype=subtype)
    code = denoise.main(["--input", str(inp), "--output", str(out)])
    return code, inp, out


@pytest.mark.parametrize("method", ["noisy_input", *METHODS])
@pytest.mark.parametrize("sr", [8000, 16000, 44100, 48000])
def test_enhance_keeps_length_and_is_finite(method, sr):
    x = noisy_speechlike(int(1.3 * sr) + 7, sr)
    y = enhance(x, sr, method, DEFAULT)
    assert y.shape == x.shape and np.isfinite(y).all()


@pytest.mark.parametrize("n", [1, 2, 100, 511, 513])
def test_very_short_input(n):
    x = noisy_speechlike(n, 16000)
    for m in METHODS:
        if m == "basic_filter" and n < 2:
            continue
        y = enhance(x, 16000, m, DEFAULT)
        assert y.shape == x.shape and np.isfinite(y).all()


def test_silence_stays_silent():
    y = enhance(np.zeros(16000), 16000, "mmse_lsa", DEFAULT)
    assert np.isfinite(y).all() and np.max(np.abs(y)) < 1e-6


def test_deterministic():
    x = noisy_speechlike(48000, 16000)
    assert np.array_equal(enhance(x, 16000, "mmse_lsa"), enhance(x, 16000, "mmse_lsa"))


def test_submission_io_contract(tmp_path):
    sr = 16000
    mono = noisy_speechlike(48000, sr)
    stereo = np.stack([noisy_speechlike(40000, sr, 1), noisy_speechlike(40000, sr, 2)], axis=1)
    loud = np.clip(4 * noisy_speechlike(30000, sr, 3), -1, 1)
    files = {
        "a_noisy.wav": (mono, sr, "PCM_16"),
        "b_stereo.wav": (stereo, sr, "PCM_16"),
        "c_loud.wav": (loud, sr, "PCM_16"),
        "d_float.wav": (mono.astype(np.float32), sr, "FLOAT"),
        "e_48k.WAV": (noisy_speechlike(70001, 48000), 48000, "PCM_24"),
        "sub/f_nested.wav": (mono[:20000], sr, "PCM_16"),
    }
    code, inp, out = run(tmp_path, files)
    (inp / "notes.txt").write_text("not audio")
    assert code == 0
    for rel in files:
        a, b = sf.info(str(inp / rel)), sf.info(str(out / rel))
        assert (a.samplerate, a.channels, a.frames, a.subtype, a.format) == \
               (b.samplerate, b.channels, b.frames, b.subtype, b.format), rel
        y, _ = sf.read(str(out / rel))
        assert np.isfinite(y).all()
    assert sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()) == \
           sorted(files)


def test_actually_reduces_noise():
    """Stationary white noise on a tone mixture: output must be closer to clean."""
    sr = 16000
    rng = np.random.default_rng(5)
    t = np.arange(48000) / sr
    clean = 0.3 * np.sin(2 * np.pi * 440 * t) * (t % 1 < 0.6)
    noisy = clean + 0.05 * rng.standard_normal(t.size)
    y = enhance(noisy, sr, "mmse_lsa")
    assert np.sum((y - clean) ** 2) < 0.5 * np.sum((noisy - clean) ** 2)
