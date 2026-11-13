"""Category-specific additions to the tracked noise PSD (noisy input only).

Each function returns delta >= 0 with the shape of the noise PSD. The caller
uses noise + weight * delta, so weight 0 leaves the base estimate untouched.

tone_delta     C1, tones and sweeps: along narrow peak tracks that also run
               through noise-dominant frames (speech harmonics do not), lift the
               noise PSD to the observed power in the peak and its main lobe.
impulse_delta  C2, clicks and claps: in short broadband bursts without harmonic
               structure, lift the noise PSD by the excess over the per-bin
               median of the neighbouring frames.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, median_filter

from . import features as ft

TONE_LOBE_BINS = 2      # half width of the Hann main lobe that is lifted with the peak


def tone_delta(power, noise, f, hop_sec):
    track, _ = ft.noise_tone_tracks(power, f, hop_sec)
    if not track.any():
        return np.zeros_like(power)
    lobe = binary_dilation(track, structure=np.ones((2 * TONE_LOBE_BINS + 1, 1), dtype=bool))
    return np.where(lobe, np.maximum(power - noise, 0.0), 0.0)


def impulse_delta(power, noise, f, hop_sec):
    burst = ft.bursts(power, f, hop_sec)
    delta = np.zeros_like(power)
    if not burst.any():
        return delta
    size = int(round(ft.BURST_MEDIAN_SEC / hop_sec)) | 1
    around = median_filter(power, size=(1, size), mode="nearest")
    delta[:, burst] = np.maximum(power[:, burst] - np.maximum(around[:, burst], noise[:, burst]),
                                 0.0)
    return delta
