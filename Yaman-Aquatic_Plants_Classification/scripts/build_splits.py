"""Write reproducible train/val/test split lists for the lab, web and combined image sets.

    python scripts/build_splits.py --data lab+web --seeds 8 9 10 11 12
    python scripts/build_splits.py --data all --seeds 8          # lab, web and lab+web

Why a script and not ``train_test_split`` inside the dataloader:

* The web images are GBIF-mediated iNaturalist observations and one observation often has
  several photographs of the same plant (up to 15). Splitting by image would put photos of
  one plant in both train and test. Here the unit of splitting is the observation
  (``gbif_occurrence_key`` from ``download_manifest.csv``); lab photographs are their own
  group.
* Splits are stratified per (source, species) and seeded per (source, species), so the lab
  partition of ``lab+web_seed8.csv`` is identical to ``lab_seed8.csv``. The "lab test" column
  is therefore the same 41 images whichever data the model was trained on.
* The 42 outdoor photographs are never split: they are all ``field_test``.
* The CSV is small, human-readable, committed to git, and records the licence and rights
  holder of every web image so the paper's data statement can be generated from it.

Columns: path (relative to the repository root, forward slashes), label, label_idx, source
(lab | web | field), group, split (train | val | test | field_test), license, rights_holder.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import config as cfg  # noqa: E402

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
FIELDNAMES = ["path", "label", "label_idx", "source", "group", "split", "license", "rights_holder"]
SPLIT_FRACTIONS = {"train": cfg.train_percent, "val": cfg.val_percent, "test": cfg.test_percent}


def canonical_class_name(folder_name: str) -> str:
    """'Brasenia schreberi (watershield)' and 'Brasenia_schreberi' -> 'Brasenia schreberi'."""
    return folder_name.split("(")[0].strip().replace("_", " ").strip()


def rel_path(path: str) -> str:
    return os.path.relpath(os.path.abspath(path), cfg.REPO_ROOT).replace("\\", "/")


def load_manifest(manifest_path):
    """basename -> (occurrence key, license, rights holder) for downloaded web images."""
    lookup = {}
    if not manifest_path or not os.path.isfile(manifest_path):
        return lookup
    with open(manifest_path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") != "downloaded" or not row.get("file_path"):
                continue
            key = os.path.basename(row["file_path"].replace("\\", "/"))
            group = row.get("gbif_occurrence_key") or row.get("observation_id") or ""
            lookup[key] = (group.strip(), (row.get("license") or "").strip(), (row.get("rights_holder") or "").strip())
    return lookup


def scan_root(root, source, manifest=None):
    """Image records for an ImageFolder-style directory. Unknown species are skipped."""
    if not root or not os.path.isdir(root):
        raise FileNotFoundError(f"{source} image folder not found: {root!r}")
    manifest = manifest or {}
    records, skipped = [], Counter()
    for entry in sorted(os.scandir(root), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        label = canonical_class_name(entry.name)
        if label not in cfg.CANONICAL_CLASS_TO_INDEX:
            skipped[label] += 1
            continue
        for file in sorted(os.scandir(entry.path), key=lambda e: e.name):
            if not file.is_file() or os.path.splitext(file.name)[1].lower() not in IMAGE_EXT:
                continue
            group, licence, holder = manifest.get(file.name, ("", "", ""))
            if not group:
                # Lab and field photographs are independent shots; the file is its own group.
                group = rel_path(file.path)
            records.append({
                "path": rel_path(file.path),
                "label": label,
                "label_idx": cfg.CANONICAL_CLASS_TO_INDEX[label],
                "source": source,
                "group": f"{source}:{group}",
                "split": "",
                "license": licence,
                "rights_holder": holder,
            })
    for label, n in skipped.items():
        print(f"  skipping folder for unknown species '{label}' under {root}")
    return records


def _class_seed(seed, source, label):
    """Deterministic per-(source, species) seed so partial configs share partitions."""
    return seed * 10_000 + {"lab": 0, "web": 5_000}.get(source, 9_000) + cfg.CANONICAL_CLASS_TO_INDEX[label]


def split_groups(groups, n_images, rng):
    """Assign observation groups to train/val/test by image-count deficit.

    ``groups`` is a list of (group_id, [records]). Guarantees: at least one group in train;
    one in test when there are two or more groups; one in val when there are three or more.
    """
    order = list(groups)
    rng.shuffle(order)
    targets = {name: max(frac * n_images, 1e-9) for name, frac in SPLIT_FRACTIONS.items()}
    counts = {name: 0 for name in SPLIT_FRACTIONS}
    assignment = {}
    minimum_order = ["train", "test", "val"]
    for i, (gid, members) in enumerate(order):
        if i < 3:
            bucket = minimum_order[i]
        else:
            # Put the group where the fill ratio after assignment stays lowest; ties -> train.
            bucket = min(
                counts,
                key=lambda name: ((counts[name] + len(members)) / targets[name], name != "train"),
            )
        assignment[gid] = bucket
        counts[bucket] += len(members)
    return assignment


def build_split(data_config, seed):
    """Records with the ``split`` column filled for one data configuration and seed."""
    if data_config not in cfg.DATA_CONFIGS:
        raise ValueError(f"data_config must be one of {cfg.DATA_CONFIGS}, got {data_config!r}")
    sources = []
    if "lab" in data_config.split("+"):
        sources.append(("lab", cfg.LAB_DATA_DIR, None))
    if "web" in data_config.split("+"):
        sources.append(("web", cfg.WEB_DATA_DIR, load_manifest(cfg.WEB_MANIFEST)))

    records = []
    for source, root, manifest in sources:
        source_records = scan_root(root, source, manifest)
        by_class = defaultdict(lambda: defaultdict(list))
        for record in source_records:
            by_class[record["label"]][record["group"]].append(record)
        for label, groups in by_class.items():
            rng = random.Random(_class_seed(seed, source, label))
            n_images = sum(len(v) for v in groups.values())
            assignment = split_groups(sorted(groups.items()), n_images, rng)
            for gid, members in groups.items():
                for record in members:
                    record["split"] = assignment[gid]
        records.extend(source_records)

    if cfg.OOD_DATA_DIR:
        for record in scan_root(cfg.OOD_DATA_DIR, "field"):
            record["split"] = "field_test"
            records.append(record)
    else:
        print("  outdoor folder not found; no field_test rows written")

    check_leakage(records)
    return records


def check_leakage(records):
    seen = defaultdict(set)
    for record in records:
        seen[record["group"]].add(record["split"])
    leaks = {g: s for g, s in seen.items() if len(s) > 1}
    if leaks:
        raise RuntimeError(f"{len(leaks)} observation group(s) span more than one split: {list(leaks)[:5]}")


def write_split_csv(records, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for record in sorted(records, key=lambda r: (r["source"], r["split"], r["label"], r["path"])):
            writer.writerow({k: record[k] for k in FIELDNAMES})


def read_split_csv(path):
    with open(path, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["label_idx"] = int(row["label_idx"])
    return rows


def summarize(records):
    table = defaultdict(Counter)
    for record in records:
        table[record["source"]][record["split"]] += 1
    print("  source     " + "".join(f"{s:>12s}" for s in ("train", "val", "test", "field_test")))
    for source in ("lab", "web", "field"):
        if source in table:
            print(f"  {source:10s} " + "".join(f"{table[source][s]:>12d}" for s in ("train", "val", "test", "field_test")))
    classes_in_train = {r["label"] for r in records if r["split"] == "train"}
    missing = [c for c in cfg.CANONICAL_CLASSNAMES_LIST if c not in classes_in_train]
    if missing:
        print(f"  WARNING: no training images for {missing}")
    n_groups = len({r["group"] for r in records if r["source"] == "web"})
    if n_groups:
        print(f"  web observations (groups): {n_groups}")


def ensure_split(data_config, seed, force=False, verbose=True):
    """Return the CSV path for (data_config, seed), building it if needed."""
    path = cfg.split_csv_path(data_config, seed)
    if os.path.isfile(path) and not force:
        return path
    if verbose:
        print(f"Building split {data_config} seed {seed} -> {path}")
    records = build_split(data_config, seed)
    write_split_csv(records, path)
    if verbose:
        summarize(records)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", choices=[*cfg.DATA_CONFIGS, "all"], default="all")
    parser.add_argument("--seeds", type=int, nargs="+", default=[8, 9, 10, 11, 12])
    parser.add_argument("--force", action="store_true", help="overwrite existing CSVs")
    args = parser.parse_args()

    configs = list(cfg.DATA_CONFIGS) if args.data == "all" else [args.data]
    for data_config in configs:
        for seed in args.seeds:
            path = cfg.split_csv_path(data_config, seed)
            if os.path.isfile(path) and not args.force:
                print(f"exists: {path}")
                continue
            ensure_split(data_config, seed, force=args.force)


if __name__ == "__main__":
    main()
