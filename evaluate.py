"""Score methods with PESQ-WB (16 kHz) and SNR, and log every run.

Scoring matches the course notebook: no level normalisation or time alignment,
plain mean over files. Clean speech is used only here, never in denoise.py.

    python evaluate.py                                     # submission config, validation
    python evaluate.py --split train --subset quick        # fast tuning set (~200 files)
    python evaluate.py --split train                       # all 1,296 train files
    python evaluate.py --split all                         # train and validation, per split
    python evaluate.py --methods noisy_input basic_filter spectral_subtraction wiener_filter mmse_lsa
    python evaluate.py --config configs/xxx.json --set lsa_gain_floor=0.1 --tag floor01
    python evaluate.py --round 2                           # submission round 2, train + validation
                                                           # -> results/submissions/round2/ (+ train/)

Each run writes results/runs/<run_id>/ (per-file CSV, breakdowns, config) and
appends one line per method to results/experiments.csv.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from nr import DEFAULT, enhance, load_config  # noqa: E402
from tools.common import fingerprint, round_dir  # noqa: E402

SUBMISSION = HERE / "configs" / "submission.json"
LOG = HERE / "results" / "experiments.csv"


REQUIRED_COLUMNS = {"segment_id", "clean_path", "noisy_path", "sample_rate_hz", "num_samples",
                    "noise_category", "noise_collection", "noise_code", "noise_difficulty",
                    "target_snr_db", "split"}


def snr_db(clean, est):
    """10 log10(sum s^2 / sum (s - s_hat)^2), no level alignment (course definition)."""
    if clean.shape != est.shape:
        raise ValueError("SNR requires equal lengths.")
    energy = np.sum(clean ** 2)
    if energy <= 0:
        return np.nan  # undefined for a silent reference
    err = np.sum((clean - est) ** 2)
    return float(np.inf if err == 0 else 10 * np.log10(energy / err))


def pesq_score(clean, est, sr):
    """(PESQ-WB, error message). Failures and invalid values become NaN, never 0."""
    from pesq import PesqError, pesq
    try:
        value = float(pesq(sr, clean, est, mode="wb", on_error=PesqError.RAISE_EXCEPTION))
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"Invalid PESQ return value: {value}")
        return value, ""
    except Exception as e:
        return np.nan, f"{type(e).__name__}: {e}"


def pesq_wb(clean, est, sr):
    return pesq_score(clean, est, sr)[0]


def read_checked(root: Path, row: dict, kind: str) -> tuple[np.ndarray, int]:
    """Read clean/noisy audio and check it against the manifest row."""
    x, sr = sf.read(str(root / row[f"{kind}_path"]), dtype="float64")
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError(f"{row['segment_id']} {kind}: expected a finite mono waveform")
    if sr != int(row["sample_rate_hz"]):
        raise ValueError(f"{row['segment_id']} {kind}: sample rate {sr} != manifest")
    if x.size != int(row["num_samples"]):
        raise ValueError(f"{row['segment_id']} {kind}: length {x.size} != manifest")
    return x, sr


def _score_one(job):
    root, row, methods, cfg = job
    clean, sr = read_checked(root, row, "clean")
    noisy, _ = read_checked(root, row, "noisy")
    # score the same valid region for every method; enhance the full waveform
    v = row.get("valid_samples")
    valid = int(v) if v is not None and pd.notna(v) else len(clean)
    if not 0 < valid <= len(clean):
        raise ValueError(f"Invalid valid_samples: {row['segment_id']}")
    out = []
    for m in methods:
        y = enhance(noisy, sr, m, cfg)
        p, err = pesq_score(clean[:valid], y[:valid], sr)
        out.append({"segment_id": row["segment_id"], "method": m,
                    "noise_category": row["noise_category"],
                    "noise_code": row["noise_code"],
                    "noise_difficulty": row["noise_difficulty"],
                    "target_snr_db": float(row["target_snr_db"]),
                    "pesq": p, "snr_db": snr_db(clean[:valid], y[:valid]), "pesq_error": err})
    return out


def summarize(detail: pd.DataFrame, order: dict) -> pd.DataFrame:
    return (detail.groupby("method")
            .agg(n=("pesq", "size"), pesq=("pesq", "mean"), pesq_std=("pesq", "std"),
                 pesq_fail=("pesq", lambda s: int(s.isna().sum())),
                 delta_pesq=("delta_pesq", "mean"), snr_db=("snr_db", "mean"),
                 snr_std=("snr_db", "std"), delta_snr_db=("delta_snr_db", "mean"))
            .sort_index(key=lambda s: s.map(order)))


def package_versions() -> dict:
    out = {}
    for name in ("numpy", "scipy", "soundfile", "pandas", "pesq"):
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def parse_overrides(items):
    out = {}
    for item in items or []:
        k, v = item.split("=", 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def git_commit():
    try:
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE,
                           capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "nr", "configs",
                                "denoise.py", "evaluate.py"], cwd=HERE,
                               capture_output=True, text=True).stdout.strip()
        return h + ("+dirty" if dirty else "")
    except Exception:
        return "no-git"


def config_label(config: Path | None, methods) -> str:
    if config is None:
        return "defaults (nr/config.py)" if methods else "configs/submission.json"
    p = config.resolve()
    return p.relative_to(HERE).as_posix() if p.is_relative_to(HERE) else str(config)


def load_manifest(root: Path, split: str, subset: str | None) -> pd.DataFrame:
    """samples.csv of a split, validated like the course notebook."""
    m = pd.read_csv(root / split / "samples.csv", encoding="utf-8-sig")
    if m.empty or not REQUIRED_COLUMNS.issubset(m.columns):
        raise ValueError(f"{split}/samples.csv: empty or missing columns "
                         f"{sorted(REQUIRED_COLUMNS - set(m.columns))}")
    if m.segment_id.duplicated().any():
        raise ValueError(f"{split}/samples.csv: duplicate segment ids")
    if not m["split"].eq(split).all():
        raise ValueError(f"{split}/samples.csv: split column mismatch")
    for col in ("clean_path", "noisy_path"):
        for rel in m[col]:
            # normpath, not resolve(): train/ and validation/ may be symlinks (Colab + Drive)
            path = Path(os.path.normpath(root / rel))
            if not path.is_relative_to(root) or not path.is_file():
                raise FileNotFoundError(f"Invalid/missing file: {path}")
    if subset:
        ids = (HERE / "splits" / f"{split}_{subset}.txt").read_text(encoding="utf-8").split()
        m = m[m.segment_id.isin(set(ids))]
    return m.reset_index(drop=True)


def evaluate_split(split, subset, methods, cfg, args, run_id, out_dir, commit):
    """Score one split, write its result folder and log it; return its summary."""
    root = args.data_root.resolve()
    manifest = load_manifest(root, split, subset)
    if args.limit:
        manifest = manifest.head(args.limit)
    jobs = [(root, r, methods, cfg) for r in manifest.to_dict("records")]

    t0 = time.time()
    with ProcessPoolExecutor(args.jobs) as ex:
        rows = [r for chunk in ex.map(_score_one, jobs, chunksize=4) for r in chunk]
    elapsed = time.time() - t0

    detail = pd.DataFrame(rows)
    base = detail[detail.method == "noisy_input"].set_index("segment_id")
    detail["delta_pesq"] = detail.pesq - detail.segment_id.map(base.pesq)
    detail["delta_snr_db"] = detail.snr_db - detail.segment_id.map(base.snr_db)

    order = {m: i for i, m in enumerate(methods)}
    summary = summarize(detail, order)
    # if PESQ failed somewhere, also compare methods on the files where all succeeded
    complete = detail.groupby("segment_id").pesq.apply(lambda x: x.notna().all())
    common = summarize(detail[detail.segment_id.isin(complete[complete].index)], order)

    out_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(out_dir / "per_file.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "summary.csv", encoding="utf-8-sig")
    if not complete.all():
        common.to_csv(out_dir / "summary_common_subset.csv", encoding="utf-8-sig")
    tables = {}
    for g in ("noise_difficulty", "target_snr_db", "noise_category"):
        tab = detail.pivot_table(index=g, columns="method", values="pesq", aggfunc="mean")
        tab = tab[[m for m in methods if m in tab.columns]]
        tab.insert(0, "n", detail[detail.method == "noisy_input"].groupby(g).size())
        tab.to_csv(out_dir / f"pesq_by_{g}.csv", encoding="utf-8-sig")
        tables[g] = tab
    run_info = {"run_id": run_id, "git_commit": commit,
                "code_fingerprint": fingerprint(HERE),
                "config_file": config_label(args.config, args.methods),
                "round": args.round, "split": split,
                "subset": subset or "all", "files": len(manifest), "methods": methods,
                "config": cfg.to_dict(), "note": args.note, "elapsed_sec": round(elapsed, 1),
                "pesq": "wb, 16 kHz, pesq==0.0.4, no normalisation",
                "snr": "10*log10(sum(clean**2)/sum((clean-output)**2)); no level alignment",
                "aggregation": "unweighted mean over files",
                "pesq_failures": int(detail.pesq.isna().sum()),
                "common_pesq_files": int(complete.sum()),
                "python": platform.python_version(), "versions": package_versions()}
    (out_dir / "run_config.json").write_text(json.dumps(run_info, indent=2, ensure_ascii=False),
                                             encoding="utf-8")

    if not args.limit:  # smoke tests are not logged
        log_rows = []
        diff = tables["noise_difficulty"]
        snr = tables["target_snr_db"]
        for m, s in summary.iterrows():
            r = {"run_id": run_id, "git_commit": commit, "split": split,
                 "subset": subset or "all", "n": int(s.n), "method": m,
                 "pesq": round(s.pesq, 4), "delta_pesq": round(s.delta_pesq, 4),
                 "snr_db": round(s.snr_db, 3), "pesq_fail": int(s.pesq_fail)}
            for lvl in ("easy", "moderate", "hard"):
                r[f"pesq_{lvl}"] = round(diff.loc[lvl, m], 4) if lvl in diff.index else np.nan
            for db in (0.0, 5.0, 10.0, 15.0):
                r[f"pesq_{int(db)}dB"] = round(snr.loc[db, m], 4) if db in snr.index else np.nan
            r["config"] = json.dumps(cfg.to_dict()) if m != "noisy_input" else ""
            r["note"] = args.note
            log_rows.append(r)
        LOG.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(log_rows).to_csv(LOG, mode="a", header=not LOG.exists(), index=False,
                                      encoding="utf-8-sig" if not LOG.exists() else "utf-8")

    pd.set_option("display.width", 200)
    print(f"\n[{split}/{subset or 'all'}] {len(manifest)} files, {elapsed:.1f}s, "
          f"commit {commit}  -> {out_dir}")
    print(summary.round(3).to_string())
    if not complete.all():
        fails = detail[detail.pesq.isna()]
        print(f"\nPESQ failed on {len(fails)} file/method pairs (see pesq_error in per_file.csv); "
              f"methods on the {int(complete.sum())} files where all succeeded:")
        print(common.round(3).to_string())
    for g in ("noise_difficulty", "target_snr_db"):
        print(f"\nPESQ by {g}\n" + tables[g].round(3).to_string())
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=HERE)
    ap.add_argument("--split", default="validation", choices=["train", "validation", "all"],
                    help="all = train (1,296) and validation (164), each scored separately")
    ap.add_argument("--subset", help="name of splits/<split>_<subset>.txt (e.g. quick)")
    ap.add_argument("--config", type=Path, help="JSON config (default: configs/submission.json)")
    ap.add_argument("--methods", nargs="+", help="methods to compare (default: config method)")
    ap.add_argument("--set", nargs="*", metavar="KEY=VALUE", help="override Config fields")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--limit", type=int, help="only the first N files (smoke test, not logged)")
    ap.add_argument("--tag", default="", help="short label for this run")
    ap.add_argument("--note", default="", help="free-text note saved in the experiment log")
    ap.add_argument("--out", type=Path, help="write results here instead of results/runs/<id>")
    ap.add_argument("--round", type=int,
                    help="submission round N: submission config on EVERY train and validation "
                         "file -> results/submissions/roundN/ (validation) and "
                         "results/submissions/roundN/train/")
    args = ap.parse_args(argv)
    if args.round is not None:
        if args.split != "validation" or args.subset or args.limit or args.set \
                or args.methods \
                or (args.config and args.config.resolve() != SUBMISSION.resolve()):
            ap.error("--round always scores configs/submission.json on every train and "
                     "validation file")
        args.config = SUBMISSION
        args.split = "all"
        args.out = args.out or round_dir(args.round)
        args.tag = args.tag or f"round{args.round}"
        args.note = args.note or f"submission round {args.round}"
    if args.split == "all" and args.subset:
        ap.error("--subset needs a single --split")

    if args.config or not args.methods:
        cfg_method, cfg = load_config(args.config or SUBMISSION)
    else:
        cfg_method, cfg = None, DEFAULT
    cfg = cfg.with_overrides(**parse_overrides(args.set))
    methods = args.methods or [cfg_method]
    methods = ["noisy_input"] + [m for m in methods if m != "noisy_input"]

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + (f"_{args.tag}" if args.tag else "")
    base_dir = args.out or (HERE / "results" / "runs" / run_id)
    base_dir = base_dir if base_dir.is_absolute() else (Path.cwd() / base_dir)
    commit = git_commit()

    if args.split != "all":
        return evaluate_split(args.split, args.subset, methods, cfg, args, run_id, base_dir,
                              commit)
    summaries = {}
    for split in ("validation", "train"):
        if args.round is not None:  # validation stays where the submission tools read it
            out_dir = base_dir if split == "validation" else base_dir / "train"
        else:
            out_dir = base_dir / split
        summaries[split] = evaluate_split(split, None, methods, cfg, args, run_id, out_dir,
                                          commit)
    both = pd.concat(summaries, names=["split"])
    both.to_csv(base_dir / "summary_by_split.csv", encoding="utf-8-sig")
    print("\n[train + validation]\n" + both[["n", "pesq", "delta_pesq", "snr_db",
                                             "delta_snr_db"]].round(3).to_string())
    return both


if __name__ == "__main__":
    main()
