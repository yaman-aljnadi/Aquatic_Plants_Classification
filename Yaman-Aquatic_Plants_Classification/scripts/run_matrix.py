"""Run the experiment grid one job at a time, skipping jobs that already have results.

    python scripts/run_matrix.py --suite data --dry-run
    python scripts/run_matrix.py --suite data            # data x head x seed on ConvNeXt-V2 Tiny
    python scripts/run_matrix.py --suite backbones       # other encoders on lab+web, linear head
    python scripts/run_matrix.py --suite curve           # learning curve on lab+web
    python scripts/run_matrix.py --suite pooled         # lab+web+field heads and learning curve
    python scripts/run_matrix.py --suite all

A job is one training run identified by (data, head, backbone, seed, train_fraction).
Every job scores the plain model and the +YNLT(224) variant in a single ``results.json``,
so YNLT rows do not need their own training. Jobs whose ``results.json`` already exists
are skipped, which makes the script safe to re-run after an interruption.

Runs are sequential: there is one 8 GB GPU.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from collections import namedtuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import config as cfg  # noqa: E402

Job = namedtuple("Job", "data head backbone seed fraction")

MAIN_BACKBONE = "convnextv2"
MAIN_DATA = "lab+web"
POOLED_DATA = "lab+web+field"
DATA_CONFIGS = list(cfg.DATA_CONFIGS)                      # lab, web, lab+web
HEADS = ["linear", "mlp", "gated_attention"]                # mlp = parameter-matched control
BASELINE_BACKBONES = ["convnext", "resnet50", "efficientnet", "vit"]
SMALL_BACKBONES = list(cfg.SMALL_CONVNEXTV2_PRESETS)        # nano, pico, atto: rover candidates
SEEDS = [8, 9, 10, 11, 12]
CURVE_FRACTIONS = [0.1, 0.25, 0.5, 1.0]


def suite_jobs(suite, seeds):
    jobs = []
    if suite in ("data", "all"):
        # Which training data and which head: the "best combination" grid.
        for data in DATA_CONFIGS:
            for head in HEADS:
                jobs += [Job(data, head, MAIN_BACKBONE, seed, 1.0) for seed in seeds]
    if suite in ("backbones", "all"):
        # Same frozen-encoder recipe and linear head, so the encoder is the only difference.
        for backbone in BASELINE_BACKBONES + SMALL_BACKBONES:
            jobs += [Job(MAIN_DATA, "linear", backbone, seed, 1.0) for seed in seeds[:3]]
    if suite in ("small", "all"):
        # Rover candidates with the full method (gated head), to see what shrinking costs.
        for backbone in SMALL_BACKBONES:
            jobs += [Job(MAIN_DATA, "gated_attention", backbone, seed, 1.0) for seed in seeds[:3]]
    if suite in ("curve", "all"):
        # Accuracy vs. number of training images, fixed val/test.
        for fraction in CURVE_FRACTIONS:
            for head in ("linear", "gated_attention"):
                jobs += [Job(MAIN_DATA, head, MAIN_BACKBONE, seed, fraction) for seed in seeds[:3]]
    if suite == "pooled":
        # One training collection: cleaned lab + web + outdoor. Handheld folder is not in the splits.
        for head in HEADS:
            jobs += [Job(POOLED_DATA, head, MAIN_BACKBONE, seed, 1.0) for seed in seeds]
        for fraction in CURVE_FRACTIONS:
            for head in ("linear", "gated_attention"):
                jobs += [Job(POOLED_DATA, head, MAIN_BACKBONE, seed, fraction) for seed in seeds[:3]]
    seen, unique = set(), []
    for job in jobs:
        if job not in seen:
            seen.add(job)
            unique.append(job)
    return unique


def completed_jobs(runs_dir):
    done = set()
    for path in glob.glob(os.path.join(runs_dir, "**", "results.json"), recursive=True):
        try:
            with open(path, encoding="utf-8") as handle:
                record = json.load(handle)
            if "data" not in record:
                continue  # archived lab-only run from the old protocol
            done.add(Job(
                record["data"],
                record["head"],
                record.get("backbone_key", record.get("backbone")),
                record["seed"],
                float(record.get("train_fraction", 1.0)),
            ))
        except (json.JSONDecodeError, KeyError):
            continue
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["data", "backbones", "small", "curve", "pooled", "all"], default="data")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--ynlt-source", default=cfg.ynlt_patch_source)
    parser.add_argument("--no-ynlt", action="store_true", help="skip the second-look evaluation")
    parser.add_argument("--force", action="store_true", help="re-run jobs that already have results.json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--runs", default=os.path.join(ROOT, "runs", "models"))
    args = parser.parse_args()

    jobs = suite_jobs(args.suite, args.seeds)
    done = set() if args.force else completed_jobs(args.runs)
    pending = [job for job in jobs if job not in done]

    print(f"Suite '{args.suite}': {len(jobs)} job(s), {len(jobs) - len(pending)} already done, {len(pending)} to run.")
    for job in jobs:
        mark = "skip" if job in done else "run "
        frac = "" if job.fraction >= 1.0 else f"  frac {job.fraction:g}"
        print(f"  [{mark}] {job.data:8s} {job.head:16s} {job.backbone:16s} seed {job.seed}{frac}")
    if not pending or args.dry_run:
        return

    failures = []
    t_start = time.time()
    for index, job in enumerate(pending, start=1):
        cmd = [
            sys.executable, os.path.join(ROOT, "train.py"),
            "--data", job.data,
            "--head", job.head,
            "--backbone", job.backbone,
            "--seed", str(job.seed),
        ]
        if job.fraction < 1.0:
            cmd += ["--train-fraction", str(job.fraction)]
        if not args.no_ynlt:
            cmd += ["--with-ynlt", "--ynlt-source", args.ynlt_source]
        for flag, value in (("--epochs", args.epochs), ("--batch-size", args.batch_size), ("--workers", args.workers)):
            if value is not None:
                cmd += [flag, str(value)]
        print(f"\n=== [{index}/{len(pending)}] {' '.join(cmd[1:])} ===", flush=True)
        result = subprocess.run(cmd, cwd=ROOT)
        if result.returncode != 0:
            print(f"FAILED (exit {result.returncode}): {job}")
            failures.append(job)

    print(f"\nMatrix finished in {(time.time() - t_start) / 60:.1f} min. {len(failures)} failure(s).")
    for job in failures:
        print(f"  failed: {job}")
    print("Aggregate with: python scripts/aggregate_results.py --latex")


if __name__ == "__main__":
    main()
