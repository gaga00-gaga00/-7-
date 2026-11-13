"""Noise-character features of one file, computed from the noisy waveform only.

STFT: Hann 1024, hop 256 (64 ms / 16 ms at 16 kHz). The 20 % lowest-energy
frames are taken as "noise-dominant".

F1 flatness     geometric / arithmetic mean of the mean noise-frame spectrum
                (low: tonal or strongly coloured noise)
F2 tone_persist share of the noise-dominant frames crossed by a noise tone: a narrow
                peak (>= 10 dB over the median smoothed spectrum) on a track that
                lasts >= 0.2 s and also runs through noise-dominant frames, which
                speech harmonics do not
F3 tone_drift   mean frequency movement along those noise-tone tracks, Hz per
                second (large: a moving tone such as a siren)
F4 impulsive    share of frames inside a short broadband burst (<= 0.1 s, >= 6 dB
                over the local median energy, no harmonic structure): clicks, claps
F5 variability  coefficient of variation over time of the energy in the
                noise-dominant band (the quarter of the bins where the
                noise-frame spectrum is closest to the whole-file spectrum)
F6 low_ratio    energy below 300 Hz / total energy of the noise-frame spectrum
"""
from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.ndimage import label, median_filter

EPS = 1e-12
N_FFT, HOP = 1024, 256
NAMES = ("flatness", "tone_persist", "tone_drift", "impulsive", "variability", "low_ratio")

NOISE_FRACTION = 0.20
PEAK_DB = 10.0          # peak height over the median-smoothed spectrum
PEAK_MEDIAN_HZ = 500.0  # width of the median smoothing along frequency
PEAK_NARROW_DB = 6.0    # drop required 4 bins (62.5 Hz) away on both sides
TRACK_SEC = 0.2         # minimum duration of a tone track
TRACK_MIN_NOISE_FRAMES = 3   # a noise tone is seen in at least this many noise frames
TRACK_MIN_NOISE_SHARE = 0.1  # ... and in this share of the frames it spans
BURST_DB = 6.0          # frame energy over the local median energy
BURST_MEDIAN_SEC = 0.4  # span of that local median
BURST_MAX_SEC = 0.1     # longer events are not impulses
BURST_MAX_PEAKS = 4.0   # mean narrow peaks per frame above this: harmonic, i.e. speech
LOW_HZ = 300.0
MIN_HZ = 50.0           # ignore DC / rumble below this in F1


def spectrogram(x, sr):
    """Power spectrogram (freq_bins, frames) in the feature STFT; n_fft scales with sr."""
    n_fft = int(round(N_FFT * sr / 16000))
    n_fft += n_fft % 2
    hop = max(1, n_fft // 4)
    x = np.asarray(x, dtype=np.float64)
    if x.size < n_fft:
        x = np.pad(x, (0, n_fft - x.size))
    f, _, z = signal.stft(x, fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop,
                          boundary=None, padded=False)
    return f, hop / sr, np.abs(z) ** 2


def noise_frames(power):
    """Indices of the lowest-energy frames."""
    energy = power.sum(axis=0)
    count = max(1, int(np.ceil(NOISE_FRACTION * power.shape[1])))
    return np.argsort(energy, kind="stable")[:count]


def tone_peaks(power, f):
    """Boolean (freq_bins, frames) mask of narrow spectral peaks, and the dB spectrogram."""
    db = 10 * np.log10(power + EPS)
    width = int(round(PEAK_MEDIAN_HZ / (f[1] - f[0]))) | 1
    base = median_filter(db, size=(width, 1), mode="nearest")
    peak = db - base >= PEAK_DB
    peak[1:-1] &= (db[1:-1] >= db[:-2]) & (db[1:-1] >= db[2:])
    peak[[0, -1]] = False
    far = np.full_like(db, -np.inf)
    far[4:-4] = np.maximum(db[:-8], db[8:])       # the louder of the bins 4 away
    peak &= db - far >= PEAK_NARROW_DB
    return peak, db


def _run_length(peak, order):
    """Length of the track ending in each peak cell; a track may move +-1 bin per frame."""
    length = np.zeros(peak.shape, dtype=np.int32)
    prev = np.zeros(peak.shape[0], dtype=np.int32)
    for t in order:
        best = prev.copy()
        best[1:] = np.maximum(best[1:], prev[:-1])
        best[:-1] = np.maximum(best[:-1], prev[1:])
        prev = np.where(peak[:, t], best + 1, 0)
        length[:, t] = prev
    return length


def tone_tracks(power, f, hop_sec):
    """Mask of peak cells that lie on a track of at least TRACK_SEC, and the dB spectrogram."""
    peak, db = tone_peaks(power, f)
    n = peak.shape[1]
    total = _run_length(peak, range(n)) + _run_length(peak, range(n - 1, -1, -1)) - 1
    return peak & (total >= max(2, int(round(TRACK_SEC / hop_sec)))), db


def noise_tone_tracks(power, f, hop_sec, noise_idx=None):
    """Tone tracks that also run through noise-dominant frames (not speech harmonics)."""
    track, db = tone_tracks(power, f, hop_sec)
    if noise_idx is None:
        noise_idx = noise_frames(power)
    lab, n = label(track, structure=np.ones((3, 3), dtype=int))
    if n == 0:
        return track, db
    n_frames = power.shape[1]
    pairs = np.unique(lab[track].astype(np.int64) * n_frames + np.nonzero(track)[1])
    ids, frames = pairs // n_frames, pairs % n_frames
    is_noise = np.zeros(n_frames, dtype=bool)
    is_noise[noise_idx] = True
    span = np.bincount(ids, minlength=n + 1)
    in_noise = np.bincount(ids, weights=is_noise[frames], minlength=n + 1)
    keep = (in_noise >= TRACK_MIN_NOISE_FRAMES) & (in_noise >= TRACK_MIN_NOISE_SHARE * span)
    keep[0] = False
    return keep[lab], db


def bursts(power, f, hop_sec, peak=None):
    """Boolean per frame: inside a short broadband burst that is not followed by harmonics."""
    if peak is None:
        peak, _ = tone_peaks(power, f)
    energy = power.sum(axis=0)
    size = int(round(BURST_MEDIAN_SEC / hop_sec)) | 1
    hot = energy >= 10 ** (BURST_DB / 10) * (median_filter(energy, size=size, mode="nearest") + EPS)
    n_peaks = peak.sum(axis=0)
    max_len = max(1, int(round(BURST_MAX_SEC / hop_sec)))
    out = np.zeros(hot.size, dtype=bool)
    edges = np.flatnonzero(np.diff(np.concatenate(([0], hot.astype(np.int8), [0]))))
    for start, stop in zip(edges[::2], edges[1::2]):
        if stop - start <= max_len and n_peaks[start:stop + 3].mean() < BURST_MAX_PEAKS:
            out[start:stop] = True
    return out


def _peak_freq(db, k, t, df):
    """Parabolic interpolation of the peak frequency around bin k of frame t."""
    a, b, c = db[k - 1, t], db[k, t], db[k + 1, t]
    den = a - 2 * b + c
    off = np.where(np.abs(den) > 1e-9, 0.5 * (a - c) / np.where(den == 0, 1, den), 0.0)
    return (k + np.clip(off, -0.5, 0.5)) * df


def tone_drift(track, db, df, hop_sec):
    """Mean |frequency change| per second between consecutive cells of the tracks."""
    moves = []
    for t in range(1, track.shape[1]):
        ks = np.flatnonzero(track[:, t])
        prev = np.flatnonzero(track[:, t - 1])
        if not len(ks) or not len(prev):
            continue
        j = np.clip(np.searchsorted(prev, ks), 1, len(prev) - 1) if len(prev) > 1 else None
        if j is None:
            near = np.full(len(ks), prev[0])
        else:
            near = np.where(np.abs(prev[j] - ks) < np.abs(prev[j - 1] - ks), prev[j], prev[j - 1])
        ok = np.abs(near - ks) <= 1
        if ok.any():
            moves.append(np.abs(_peak_freq(db, ks[ok], t, df) - _peak_freq(db, near[ok], t - 1, df)))
    return float(np.concatenate(moves).mean() / hop_sec) if moves else 0.0


def spectrum_features(x, sr):
    """(F1 flatness, F6 low_ratio) only: the two features the adaptive blend uses."""
    f, _, power = spectrogram(x, sr)
    noise = power[:, noise_frames(power)].mean(axis=1) + EPS
    band = f >= MIN_HZ
    flat = float(np.exp(np.mean(np.log(noise[band]))) / np.mean(noise[band]))
    return flat, float(noise[f <= LOW_HZ].sum() / noise.sum())


def features(x, sr) -> dict:
    """The six features of a mono waveform as a dict (see the module docstring)."""
    f, hop_sec, power = spectrogram(x, sr)
    df = f[1] - f[0]
    idx = noise_frames(power)
    noise = power[:, idx].mean(axis=1) + EPS
    band = f >= MIN_HZ

    flat = float(np.exp(np.mean(np.log(noise[band]))) / np.mean(noise[band]))

    track, db = noise_tone_tracks(power, f, hop_sec, idx)
    persist = float(track[:, idx].any(axis=0).mean())
    drift = tone_drift(track, db, df, hop_sec)

    impulsive = float(bursts(power, f, hop_sec).mean())

    # band where the noise-frame spectrum is closest to the whole-file spectrum,
    # i.e. where speech adds the least
    ratio = noise / (power.mean(axis=1) + EPS)
    ratio[~band] = -np.inf
    sel = np.argsort(ratio)[-max(1, band.sum() // 4):]
    e = power[sel].sum(axis=0)
    variability = float(e.std() / (e.mean() + EPS))

    low = float(noise[f <= LOW_HZ].sum() / noise.sum())
    return dict(zip(NAMES, (flat, persist, drift, impulsive, variability, low)))
