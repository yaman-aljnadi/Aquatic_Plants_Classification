"""Dataloaders.

Two ways to get data:

* ``get_split_loaders(cfg, split_csv)`` reads a split list written by
  ``scripts/build_splits.py`` (lab, web or lab+web, observation-grouped). This is what
  ``train.py`` uses.
* ``get_3_dataloaders`` / ``get_ood_dataloaders`` are the original ImageFolder path with an
  on-the-fly stratified split, kept for the archived lab-only runs and ``evaluate.py``
  without a split CSV. Folder names like 'Brasenia schreberi (watershield)' are remapped.
"""

import csv
import os
import random
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets


def canonical_class_name(name: str) -> str:
    return name.split("(")[0].strip()


def remap_imagefolder_to_canonical(dataset, cfg):
    remapped = []
    skipped = 0
    for path, local_idx in dataset.samples:
        local_name = canonical_class_name(dataset.classes[local_idx])
        if local_name in cfg.CANONICAL_CLASS_TO_INDEX:
            remapped.append((path, cfg.CANONICAL_CLASS_TO_INDEX[local_name]))
        else:
            skipped += 1
            print(f"Warning: skipping unknown class '{local_name}': {path}")
    if not remapped:
        raise ValueError("No images matched the canonical class list.")
    if skipped:
        print(f"Skipped {skipped} images.")
    dataset.samples = remapped
    dataset.imgs = remapped
    dataset.targets = [s[1] for s in remapped]
    dataset.classes = cfg.CANONICAL_CLASSNAMES_LIST
    dataset.class_to_idx = cfg.CANONICAL_CLASS_TO_INDEX
    return dataset


def _worker_init(_worker_id):
    """One compute thread per worker.

    The v2 transforms run torch ops inside each worker; with the default thread count
    eight workers on a 24-thread CPU oversubscribe it badly, and OpenBLAS in numpy
    allocates a per-thread buffer set per process, which has run out of memory and
    hung the loader. Both are avoided by making each worker single-threaded.
    """
    torch.set_num_threads(1)
    try:
        import cv2

        cv2.setNumThreads(0)
    except ImportError:
        pass


def _loader_kwargs(cfg, shuffle=False, workers=None):
    workers = int(getattr(cfg, "num_workers", 0)) if workers is None else int(workers)
    kwargs = {
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": bool(getattr(cfg, "pin_memory", False) and torch.cuda.is_available()),
    }
    if workers > 0:
        # Without persistent workers Windows re-spawns (and re-imports torch in) every
        # worker at the start of every epoch, which dominated the old two-hour runs.
        kwargs["persistent_workers"] = bool(getattr(cfg, "persistent_workers", True))
        kwargs["prefetch_factor"] = int(getattr(cfg, "prefetch_factor", 2))
        kwargs["worker_init_fn"] = _worker_init
    return kwargs


# --------------------------------------------------------------------------------------
# Split-list datasets (data/splits/*.csv written by scripts/build_splits.py)
# --------------------------------------------------------------------------------------
class ManifestDataset(Dataset):
    """Images listed in split records. Exposes ``samples`` so YNLT can re-open files."""

    def __init__(self, records, cfg, transform=None):
        # Keep only plain data (no module reference): Windows DataLoader workers pickle this.
        self.records = list(records)
        self.repo_root = cfg.REPO_ROOT
        self.transform = transform
        self.samples = [(self._abs(r["path"]), int(r["label_idx"])) for r in self.records]
        self.imgs = self.samples
        self.targets = [label for _, label in self.samples]
        self.classes = list(cfg.CANONICAL_CLASSNAMES_LIST)
        self.class_to_idx = dict(cfg.CANONICAL_CLASS_TO_INDEX)

    def _abs(self, path):
        return path if os.path.isabs(path) else os.path.join(self.repo_root, path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def read_split_records(split_csv):
    with open(split_csv, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["label_idx"] = int(row["label_idx"])
    return rows


def subsample_train_groups(records, fraction, seed):
    """Keep about ``fraction`` of the training images per class, whole groups at a time.

    Used for the learning curve. Every class keeps at least one group so no species
    disappears from training.
    """
    if fraction is None or fraction >= 1.0:
        return list(records)
    rng = random.Random(seed * 7919 + int(round(fraction * 1000)))
    by_class = defaultdict(lambda: defaultdict(list))
    for record in records:
        by_class[record["label_idx"]][record["group"]].append(record)
    kept = []
    for label in sorted(by_class):
        groups = sorted(by_class[label].items())
        rng.shuffle(groups)
        target = fraction * sum(len(m) for _, m in groups)
        taken = 0
        for i, (_, members) in enumerate(groups):
            if i > 0 and taken >= target:
                break
            kept.extend(members)
            taken += len(members)
    return kept


def test_record_groups(test_records):
    """Held-out loaders: per-source test rows, plus the pooled test when sources were mixed.

    Outdoor photographs are ``field_test`` in the older lists and ``source=field, split=test``
    once they are pooled into training (``lab+web+field``).
    """
    return {
        "lab_test": [r for r in test_records if r["split"] == "test" and r["source"] == "lab"],
        "web_test": [r for r in test_records if r["split"] == "test" and r["source"] == "web"],
        "field_test": [
            r for r in test_records
            if r["split"] == "field_test" or (r["split"] == "test" and r["source"] == "field")
        ],
        "pooled_test": [r for r in test_records if r["split"] == "test"],
    }


def get_split_loaders(cfg, split_csv, batch_size=None, train_fraction=None, seed=0, test_csv=None):
    """DataLoaders for one split CSV.

    ``train`` and ``val`` come from ``split_csv``. The test loaders (``lab_test``,
    ``web_test``: source-specific rows of the ``test`` split; ``field_test``: the outdoor
    photographs) come from ``test_csv`` when given, otherwise from ``split_csv``. Passing
    the lab+web list of the same seed as ``test_csv`` scores a lab-only or web-only model
    on all three test sets; this is leak-free because the per-source partitions are seeded
    identically, so a lab-only model's training rows never overlap the lab+web test rows.
    Missing splits map to ``None``. Per-loader image counts are returned under ``sizes``.
    """
    batch_size = batch_size or cfg.batch_size
    records = read_split_records(split_csv)
    test_records = read_split_records(test_csv) if test_csv and test_csv != split_csv else records
    train_records = subsample_train_groups([r for r in records if r["split"] == "train"], train_fraction, seed)
    val_records = [r for r in records if r["split"] == "val"]
    tests = test_record_groups(test_records)
    if not train_records:
        raise ValueError(f"No training rows in {split_csv}")
    if test_records is not records:
        train_paths = {r["path"] for r in records if r["split"] in ("train", "val")}
        for name, recs in tests.items():
            overlap = train_paths.intersection(r["path"] for r in recs)
            if overlap:
                raise RuntimeError(f"{len(overlap)} {name} image(s) also appear in train/val of {split_csv}")

    n_workers = int(getattr(cfg, "num_workers", 0))
    eval_workers = min(n_workers, int(getattr(cfg, "eval_workers", 2)))

    def make(recs, transform, shuffle=False, workers=None):
        if not recs:
            return None
        return DataLoader(
            ManifestDataset(recs, cfg, transform),
            batch_size=batch_size,
            **_loader_kwargs(cfg, shuffle=shuffle, workers=workers),
        )

    # Only the training loader does augmentation; evaluation loaders need few workers,
    # and every persistent worker is a 600 MB process on Windows.
    loaders = {
        "train": make(train_records, cfg.aug_tf, shuffle=True),
        "val": make(val_records, cfg.basic_tf, workers=eval_workers),
    }
    for name, recs in tests.items():
        loaders[name] = make(recs, cfg.basic_tf, workers=eval_workers)
    loaders["sizes"] = {
        "train": len(train_records),
        "val": len(val_records),
        **{name: len(recs) for name, recs in tests.items()},
    }
    print(
        "Split  " + "  ".join(f"{k}={v}" for k, v in loaders["sizes"].items())
        + (f"  (train fraction {train_fraction})" if train_fraction and train_fraction < 1.0 else "")
    )
    return loaders


class PlantData:
    def __init__(self, config, data_dir=None, batch_size=None, train_percent=None,
                 val_percent=None, test_percent=None, random_seed=None):
        self.config = config
        self.data_dir = data_dir or config.main_data_dir
        self.batch_size = batch_size if batch_size is not None else config.batch_size
        self.train_percent = train_percent if train_percent is not None else config.train_percent
        self.val_percent = val_percent if val_percent is not None else config.val_percent
        self.test_percent = test_percent if test_percent is not None else config.test_percent
        self.random_seed = random_seed if random_seed is not None else config.random_seed
        if not self.data_dir:
            raise FileNotFoundError("Lab data folder not found. Set AQUATIC_LAB_DATA or place images in 'Aquatic Plant lab'.")
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        self.dataset_init = remap_imagefolder_to_canonical(datasets.ImageFolder(self.data_dir), config)
        self.all_indices = list(range(len(self.dataset_init)))
        self.all_targets = self.dataset_init.targets

    def get_dataloaders(self):
        if self.test_percent == 1.0:
            test_ds = remap_imagefolder_to_canonical(
                datasets.ImageFolder(self.data_dir, transform=self.config.basic_tf), self.config
            )
            self.test_dataset = test_ds
            loader = DataLoader(test_ds, batch_size=self.batch_size, **_loader_kwargs(self.config))
            return None, None, loader

        train_val_idx, test_idx, train_val_y, _ = train_test_split(
            self.all_indices, self.all_targets,
            test_size=self.test_percent, stratify=self.all_targets, random_state=self.random_seed,
        )
        relative_val = self.val_percent / (self.train_percent + self.val_percent)
        train_idx, val_idx, _, _ = train_test_split(
            train_val_idx, train_val_y, test_size=relative_val,
            stratify=train_val_y, random_state=self.random_seed,
        )
        train_ds = remap_imagefolder_to_canonical(
            datasets.ImageFolder(self.data_dir, transform=self.config.aug_tf), self.config
        )
        val_ds = remap_imagefolder_to_canonical(
            datasets.ImageFolder(self.data_dir, transform=self.config.basic_tf), self.config
        )
        test_ds = remap_imagefolder_to_canonical(
            datasets.ImageFolder(self.data_dir, transform=self.config.basic_tf), self.config
        )
        self.train_dataset = Subset(train_ds, train_idx)
        self.val_dataset = Subset(val_ds, val_idx)
        self.test_dataset = Subset(test_ds, test_idx)
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, **_loader_kwargs(self.config, shuffle=True))
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, **_loader_kwargs(self.config))
        test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, **_loader_kwargs(self.config))
        print(
            f"Lab split  total={len(self.dataset_init)}  "
            f"train={len(self.train_dataset)}  val={len(self.val_dataset)}  test={len(self.test_dataset)}"
        )
        return train_loader, val_loader, test_loader


def get_3_dataloaders(cfg, data_dir=None, batch_size=None, train_percent=None,
                      val_percent=None, test_percent=None, random_seed=None):
    plant = PlantData(cfg, data_dir, batch_size, train_percent, val_percent, test_percent, random_seed)
    return plant.get_dataloaders()


def get_ood_dataloaders(cfg, ood_data_dir=None, test_percent=0.5, batch_size=None, random_seed=None):
    ood_data_dir = ood_data_dir or cfg.hand_test_data_dir
    if not ood_data_dir:
        raise FileNotFoundError("OOD data folder not found. Set AQUATIC_OOD_DATA or place images in 'Aquatic Plant outdoor'.")
    random_seed = cfg.random_seed if random_seed is None else random_seed
    batch_size = cfg.batch_size if batch_size is None else batch_size
    raw = remap_imagefolder_to_canonical(datasets.ImageFolder(ood_data_dir), cfg)
    samples = raw.samples
    indices = list(range(len(samples)))
    targets = [s[1] for s in samples]
    try:
        val_idx, test_idx, _, _ = train_test_split(
            indices, targets, test_size=test_percent, stratify=targets, random_state=random_seed
        )
    except ValueError:
        print("OOD stratify failed (a class has too few images). Using an unstratified split.")
        val_idx, test_idx, _, _ = train_test_split(
            indices, targets, test_size=test_percent, random_state=random_seed
        )
    base = datasets.ImageFolder(ood_data_dir, transform=cfg.basic_tf)
    base.samples = samples
    base.imgs = samples
    base.targets = targets
    base.classes = cfg.CANONICAL_CLASSNAMES_LIST
    base.class_to_idx = cfg.CANONICAL_CLASS_TO_INDEX
    val_ds = Subset(base, val_idx)
    test_ds = Subset(base, test_idx)
    print(f"OOD split  val={len(val_ds)}  test={len(test_ds)}")
    kw = _loader_kwargs(cfg)
    return DataLoader(val_ds, batch_size=batch_size, **kw), DataLoader(test_ds, batch_size=batch_size, **kw)


def loader_samples(loader):
    """(path, label) pairs behind a DataLoader, in dataset order, or None if unavailable.

    YNLT needs the source file to patch the full-resolution image, which the tensors
    coming out of the loader no longer contain.
    """
    dataset = getattr(loader, "dataset", loader)
    if isinstance(dataset, Subset):
        base, indices = dataset.dataset, list(dataset.indices)
    else:
        base, indices = dataset, None
    samples = getattr(base, "samples", None)
    if not samples:
        return None
    if indices is None:
        return list(samples)
    return [samples[i] for i in indices]


def load_original_image(path):
    return Image.open(path).convert("RGB")


def image_to_tensor(cfg, image, device=None):
    if isinstance(image, Image.Image):
        if image.mode != "RGB":
            image = image.convert("RGB")
        tensor = cfg.basic_tf(image)
    elif isinstance(image, np.ndarray):
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        tensor = cfg.basic_tf(Image.fromarray(image))
    elif isinstance(image, torch.Tensor):
        tensor = image
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
    else:
        raise ValueError("Input must be PIL, numpy, or tensor.")
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return tensor.to(device)


def to_uint8_img(cfg, tensor_image):
    mean = torch.tensor(cfg.normalization_parameters["mean"], device=tensor_image.device).view(3, 1, 1)
    std = torch.tensor(cfg.normalization_parameters["std"], device=tensor_image.device).view(3, 1, 1)
    img = torch.clamp(tensor_image * std + mean, 0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    return (img * 255).astype(np.uint8)
