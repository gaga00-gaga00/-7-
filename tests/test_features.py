"""Noise-character features on synthetic signals."""
import numpy as np
from scipy import signal

from nr.features import NAMES, features

SR = 16000
T = np.arange(3 * SR) / SR


def _noise(seed=0, level=0.05):
    return level * np.random.default_rng(seed).standard_normal(T.size)


def test_all_features_finite_and_named():
    for x in (_noise(), np.zeros(SR), _noise()[:300]):
        out = features(x, SR)
        assert tuple(out) == NAMES
        assert all(np.isfinite(v) for v in out.values())


def test_white_noise_is_flat_without_tones():
    out = features(_noise(), SR)
    assert out["flatness"] > 0.8
    assert out["tone_persist"] < 0.05
    assert out["low_ratio"] < 0.1


def test_steady_tone_is_persistent_and_does_not_move():
    out = features(_noise() + 0.2 * np.sin(2 * np.pi * 1000 * T), SR)
    assert out["tone_persist"] > 0.9
    assert out["tone_drift"] < 20
    assert out["flatness"] < features(_noise(), SR)["flatness"]


def test_sweep_moves():
    sweep = 0.2 * signal.chirp(T, f0=600, t1=T[-1], f1=1500)   # 300 Hz/s
    out = features(_noise() + sweep, SR)
    assert out["tone_persist"] > 0.5
    assert 150 < out["tone_drift"] < 600


def test_clicks_are_impulsive():
    x = _noise()
    y = x.copy()
    for start in (8000, 20000, 33000):                    # 2 ms broadband clicks
        y[start:start + 32] += 0.8 * np.sign(_noise(seed=1)[:32])
    assert features(x, SR)["impulsive"] == 0
    assert features(y, SR)["impulsive"] > 0.03


def test_harmonic_bursts_are_not_noise_tones_or_impulses():
    """A speech-like signal: short harmonic bursts with a gliding pitch between pauses."""
    x = _noise(level=0.01)
    for start in (0.3, 1.2, 2.1):
        seg = (T >= start) & (T < start + 0.5)
        f0 = 120 + 60 * (T[seg] - start)
        phase = 2 * np.pi * np.cumsum(f0) / SR
        x[seg] += sum(0.1 / h * np.sin(h * phase) for h in range(1, 20))
    out = features(x, SR)
    assert out["tone_persist"] < 0.05
    assert out["impulsive"] == 0


def test_fluctuating_noise_varies_and_rumble_is_low():
    x = _noise()
    am = x * (1 + 0.9 * np.sin(2 * np.pi * 1.0 * T))
    assert features(am, SR)["variability"] > 2 * features(x, SR)["variability"]
    sos = signal.butter(4, 200, btype="lowpass", fs=SR, output="sos")
    rumble = signal.sosfilt(sos, 20 * x)
    assert features(x + rumble, SR)["low_ratio"] > 0.7
