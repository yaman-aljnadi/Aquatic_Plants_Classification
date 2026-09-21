"""Run the experiment matrix the advisor asked for, one job at a time.

    python scripts/run_matrix.py --suite ablation --dry-run
    python scripts/run_matrix.py --suite all

A job is one training run identified by (head, backbone, seed). Each job already
produces the plain, +YNLT(full-res) and +YNLT(224) numbers in a single ``results.json``,
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

MAIN_BACKBONE = "convnextv2"
BASELINE_BACKBONES = ["convnext", "resnet50", "efficientnet", "vit"]
SEEDS = [8, 9, 10, 11, 12]


def suite_jobs(suite, seeds):
    """(head, backbone, seed) triples."""
    jobs = []
    if suite in ("ablation", "all"):
        # Table 1: linear vs gated attention on the shared ConvNeXt-V2 backbone, seed 8 only.
        jobs += [("linear", MAIN_BACKBONE, seeds[0]), ("gated_attention", MAIN_BACKBONE, seeds[0])]
    if suite in ("backbones", "all"):
        # Same frozen-encoder recipe, linear head, so the backbone is the only difference.
        jobs += [("linear", backbone, seeds[0]) for backbone in BASELINE_BACKBONES]
    if suite in ("seeds", "all"):
        # Repeated stratified splits (not 5-fold: some species have two lab images).
        jobs += [("gated_attention", MAIN_BACKBONE, seed) for seed in seeds]
        jobs += [("linear", MAIN_BACKBONE, seed) for seed in seeds]
    # Deduplicate, keep order.
    seen, unique = set(), []
    for job in jobs:
        if job not in seen:
            seen.add(job)
            unique.append(job)
    return unique


def completed_jobs(runs_dir):
    done = set()
    for path in glob.glob(os.path.join(runs_dir, "*", "results.json")):
        try:
            with open(path, encoding="utf-8") as handle:
                record = json.load(handle)
            done.add((record["head"], record.get("backbone_key", record.get("backbone")), record["seed"]))
        except (json.JSONDecodeError, KeyError):
            continue
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["ablation", "backbones", "seeds", "all"], default="all")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--ynlt-source", default="both")
    parser.add_argument("--no-ynlt", action="store_true", help="skip the second-look evaluation")
    parser.add_argument("--force", action="store_true", help="re-run jobs that already have results.json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--runs", default=os.path.join(ROOT, "runs", "models"))
    args = parser.parse_args()

    jobs = suite_jobs(args.suite, args.seeds)
    done = set() if args.force else completed_jobs(args.runs)
    pending = [job for job in jobs if job not in done]

    print(f"Suite '{args.suite}': {len(jobs)} job(s), {len(jobs) - len(pending)} already done, {len(pending)} to run.")
    for head, backbone, seed in jobs:
        mark = "skip" if (head, backbone, seed) in done else "run "
        print(f"  [{mark}] {head:16s} {backbone:12s} seed {seed}")
    if not pending or args.dry_run:
        return

    failures = []
    t_start = time.time()
    for index, (head, backbone, seed) in enumerate(pending, start=1):
        cmd = [
            sys.executable,
            os.path.join(ROOT, "train.py"),
            "--head", head,
            "--backbone", backbone,
            "--seed", str(seed),
        ]
        if not args.no_ynlt:
            cmd += ["--with-ynlt", "--ynlt-source", args.ynlt_source]
        for flag, value in (("--epochs", args.epochs), ("--batch-size", args.batch_size), ("--workers", args.workers)):
            if value is not None:
                cmd += [flag, str(value)]
        print(f"\n=== [{index}/{len(pending)}] {' '.join(cmd[1:])} ===", flush=True)
        result = subprocess.run(cmd, cwd=ROOT)
        if result.returncode != 0:
            print(f"FAILED (exit {result.returncode}): {head} {backbone} seed {seed}")
            failures.append((head, backbone, seed))

    print(f"\nMatrix finished in {(time.time() - t_start) / 60:.1f} min. {len(failures)} failure(s).")
    for job in failures:
        print(f"  failed: {job}")
    print("Aggregate with: python scripts/aggregate_results.py --latex")


if __name__ == "__main__":
    main()
