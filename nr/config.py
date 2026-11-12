"""Algorithm parameters and JSON config loading."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path


@dataclass(frozen=True)
class Config:
    # STFT, in seconds so other sample rates work too (512 / 128 samples @ 16 kHz)
    win_sec: float = 0.032
    hop_sec: float = 0.008
    # basic filter
    filter_order: int = 4
    lowpass_hz: float = 3500.0
    highpass_hz: float = 100.0
    bandstop_hz: tuple = (950.0, 1050.0)
    # noise estimate (course baseline: mean of the lowest-energy frames)
    noise_fraction: float = 0.20
    # spectral subtraction
    subtraction_alpha: float = 1.5
    subtraction_floor: float = 0.02
    # Wiener / decision-directed a-priori SNR
    dd_alpha: float = 0.98
    wiener_gain_floor: float = 0.05
    # MMSE-LSA
    lsa_xi_min_db: float = -25.0
    lsa_gain_floor: float = 0.05

    def with_overrides(self, **overrides) -> "Config":
        known = {f.name for f in fields(self)}
        unknown = set(overrides) - known
        if unknown:
            raise KeyError(f"Unknown Config field(s): {sorted(unknown)}")
        fixed = {k: tuple(v) if isinstance(v, list) else v for k, v in overrides.items()}
        return replace(self, **fixed)

    def to_dict(self) -> dict:
        return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}


DEFAULT = Config()


def load_config(path: str | Path) -> tuple[str, Config]:
    """Read {"method": ..., "params": {...}} and return (method, Config)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["method"], DEFAULT.with_overrides(**data.get("params", {}))
