"""GPU-resident data pipeline.

The CPU DataLoader path (``src/data.py``) decodes the 1500x1000 and ~1536-px JPEGs and
runs every augmentation in Python worker processes on each epoch; on this machine that
kept the RTX 3070 Ti at about 5 % utilisation. This module instead:

1. decodes every image **once**, squashes it to ``cfg.gpu_cache_size`` x ``gpu_cache_size``
   (320 by default; the evaluation transform ``Resize((224, 224))`` squashes too) and keeps
   it as a uint8 tensor in ``data/cache/images_<size>.pt`` (about 0.3 MB per image);
2. loads the whole bank onto the GPU (1,385 images = 425 MB) and gathers mini-batches
   with a single index;
3. applies the training augmentation as batched GPU kernels with independent random
   parameters per image (``gpu_augment``), reproducing ``cfg.aug_tf``:
   RandomResizedCrop(224, scale 0.1-1) + horizontal/vertical flips + RandomRotation(90)
   + RandomPerspective(0.15, p=0.5) are composed into **one** per-image homography and a
   single ``grid_sample``; ColorJitter(0.3, 0.3, 0.3, 0.1) and TrivialAugmentWide are
   per-image tensor arithmetic; Normalize; RandomErasing(p=0.5).

Deviations from the torchvision pipeline, all minor: geometric ops are sampled directly as
the backward map; bilinear instead of bicubic resampling; ColorJitter sub-ops are applied in
a fixed order; TrivialAugmentWide's ``Equalize`` is replaced by ``AutoContrast``; the
training crop is cut from the 320-px squashed image instead of the original file.

``get_gpu_splits`` mirrors ``data.get_split_loaders`` and returns ``GpuSplit`` objects that
``train.py``, ``metrics.evaluate_loader`` and ``ynlt.evaluate_with_ynlt`` can iterate.
"""

from __future__ import annotations

import hashlib
import math
import os
import time

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.data import read_split_records, subsample_train_groups, test_record_groups

# ----------------------------------------------------------------------------------------
# Image bank (decode once, cache on disk, live on the GPU)
# ----------------------------------------------------------------------------------------
class _DecodeDataset(Dataset):
    def __init__(self, paths, size):
        self.paths, self.size = paths, size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = Image.open(self.paths[index])
        image.draft("RGB", (self.size * 2, self.size * 2))  # JPEG DCT downscale: much faster decode
        image = image.convert("RGB").resize((self.size, self.size), Image.BILINEAR)
        array = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
        return array.view(self.size, self.size, 3).permute(2, 0, 1).contiguous(), index


def _bank_key(paths):
    digest = hashlib.sha1("\n".join(sorted(paths)).encode("utf-8")).hexdigest()
    return digest[:12]


def _cache_path(cfg, paths, size):
    cache_dir = getattr(cfg, "CACHE_DIR", os.path.join(cfg.PROJECT_ROOT, "data", "cache"))
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"images_{size}_{_bank_key(paths)}.pt")


class ImageBank:
    """All images of a split list as one uint8 tensor [N, 3, S, S] on ``device``."""

    def __init__(self, cfg, paths, device, size=None, workers=None, verbose=True):
        self.size = int(size or getattr(cfg, "gpu_cache_size", 320))
        self.paths = list(paths)
        self.index = {path: i for i, path in enumerate(self.paths)}
        cache = _cache_path(cfg, self.paths, self.size)
        if os.path.exists(cache):
            blob = torch.load(cache, map_location="cpu")
            if blob["paths"] != self.paths:
                blob = None
        else:
            blob = None
        if blob is None:
            blob = self._build(cfg, workers, verbose)
            torch.save(blob, cache)
            if verbose:
                print(f"Image bank cached: {cache}")
        elif verbose:
            print(f"Image bank loaded: {cache}  ({len(self.paths)} images at {self.size}x{self.size})")
        self.images = blob["images"].to(device, non_blocking=True)
        self.device = device

    def _build(self, cfg, workers, verbose):
        t0 = time.time()
        workers = int(getattr(cfg, "num_workers", 8) if workers is None else workers)
        loader = DataLoader(
            _DecodeDataset(self.paths, self.size), batch_size=32, num_workers=workers,
            shuffle=False, persistent_workers=False,
        )
        images = torch.empty(len(self.paths), 3, self.size, self.size, dtype=torch.uint8)
        done = 0
        for batch, indices in loader:
            images[indices] = batch
            done += batch.size(0)
            if verbose and done % 320 == 0:
                print(f"  decoded {done}/{len(self.paths)}")
        if verbose:
            print(f"Image bank built: {len(self.paths)} images in {time.time() - t0:.0f} s")
        return {"paths": self.paths, "images": images}

    def rows(self, paths):
        return torch.tensor([self.index[p] for p in paths], dtype=torch.long, device=self.device)


# ----------------------------------------------------------------------------------------
# Batched GPU transforms
# ----------------------------------------------------------------------------------------
OUT = 224


def _normalize(cfg, x):
    mean = torch.tensor(cfg.normalization_parameters["mean"], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(cfg.normalization_parameters["std"], device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


@torch.no_grad()
def gpu_eval_transform(cfg, x_uint8):
    """``basic_tf`` on the GPU: squash to 224x224, scale to [0, 1], normalise."""
    x = x_uint8.float().div_(255.0)
    x = F.interpolate(x, size=(OUT, OUT), mode="bilinear", antialias=True, align_corners=False)
    return _normalize(cfg, x)


def _homography_from_corners(dst, src):
    """Per-sample 3x3 H with H @ [dst, 1] ~ [src, 1] for four point pairs. dst, src: [B, 4, 2]."""
    B = dst.size(0)
    x, y = dst[..., 0], dst[..., 1]
    u, v = src[..., 0], src[..., 1]
    zeros, ones = torch.zeros_like(x), torch.ones_like(x)
    rows_u = torch.stack([x, y, ones, zeros, zeros, zeros, -u * x, -u * y], dim=-1)
    rows_v = torch.stack([zeros, zeros, zeros, x, y, ones, -v * x, -v * y], dim=-1)
    A = torch.cat([rows_u, rows_v], dim=1)                      # [B, 8, 8]
    b = torch.cat([u, v], dim=1).unsqueeze(-1)                  # [B, 8, 1]
    h = torch.linalg.solve(A, b).squeeze(-1)                    # [B, 8]
    return torch.cat([h, torch.ones(B, 1, device=h.device)], dim=1).view(B, 3, 3)


def _rot(theta):
    c, s = torch.cos(theta), torch.sin(theta)
    z, o = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack([torch.stack([c, -s, z], -1), torch.stack([s, c, z], -1), torch.stack([z, z, o], -1)], -2)


def _affine(a, b, c, d, tx, ty):
    z, o = torch.zeros_like(a), torch.ones_like(a)
    return torch.stack([torch.stack([a, b, tx], -1), torch.stack([c, d, ty], -1), torch.stack([z, z, o], -1)], -2)


def _sample_geometry(B, device, ta_op, ta_mag):
    """Compose crop, flips, rotation, perspective and the geometric TrivialAugment ops into
    one homography per image mapping normalised output coords to normalised input coords."""
    one = torch.ones(B, device=device)
    zero = torch.zeros(B, device=device)

    # RandomResizedCrop(scale=(0.1, 1), ratio=(3/4, 4/3)); the input is square (S x S).
    area = torch.empty(B, device=device).uniform_(0.1, 1.0)
    log_ratio = torch.empty(B, device=device).uniform_(math.log(3 / 4), math.log(4 / 3))
    ratio = torch.exp(log_ratio)
    w = torch.sqrt(area * ratio).clamp(max=1.0)
    h = torch.sqrt(area / ratio).clamp(max=1.0)
    cx = (torch.rand(B, device=device) * 2 - 1) * (1 - w)
    cy = (torch.rand(B, device=device) * 2 - 1) * (1 - h)
    crop = _affine(w, zero, zero, h, cx, cy)

    # Flips as sign changes.
    fx = torch.where(torch.rand(B, device=device) < 0.5, -one, one)
    fy = torch.where(torch.rand(B, device=device) < 0.5, -one, one)
    flip = _affine(fx, zero, zero, fy, zero, zero)

    # RandomRotation(degrees=90): angle in [-90, 90].
    theta = torch.empty(B, device=device).uniform_(-math.pi / 2, math.pi / 2)
    rot = _rot(theta)

    # RandomPerspective(distortion_scale=0.15, p=0.5): corners move inward by U(0, 0.15).
    corners = torch.tensor([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]], device=device).expand(B, 4, 2)
    jitter = torch.rand(B, 4, 2, device=device) * 0.15
    moved = corners - torch.sign(corners) * jitter
    apply = (torch.rand(B, 1, 1, device=device) < 0.5).float()
    src = corners * (1 - apply) + moved * apply
    persp = _homography_from_corners(corners, src)

    # Geometric TrivialAugmentWide ops (others are colour ops, applied later).
    sign = torch.where(torch.rand(B, device=device) < 0.5, -one, one)
    shear = 0.99 * ta_mag * sign
    shift = (32.0 / OUT) * 2 * ta_mag * sign          # TA translates by up to 32 px of 224
    ta_theta = math.radians(135.0) * ta_mag * sign
    ta = torch.eye(3, device=device).expand(B, 3, 3).clone()
    ta = torch.where((ta_op == 1).view(B, 1, 1), _affine(one, shear, zero, one, zero, zero), ta)   # ShearX
    ta = torch.where((ta_op == 2).view(B, 1, 1), _affine(one, zero, shear, one, zero, zero), ta)   # ShearY
    ta = torch.where((ta_op == 3).view(B, 1, 1), _affine(one, zero, zero, one, shift, zero), ta)   # TranslateX
    ta = torch.where((ta_op == 4).view(B, 1, 1), _affine(one, zero, zero, one, zero, shift), ta)   # TranslateY
    ta = torch.where((ta_op == 5).view(B, 1, 1), _rot(ta_theta), ta)                               # Rotate

    return crop @ flip @ rot @ persp @ ta


def _warp(x, H):
    B = x.size(0)
    lin = torch.linspace(-1, 1, OUT, device=x.device)
    gy, gx = torch.meshgrid(lin, lin, indexing="ij")
    base = torch.stack([gx, gy, torch.ones_like(gx)], dim=-1).view(1, OUT * OUT, 3)
    pts = base @ H.transpose(1, 2)                                   # [B, HW, 3]
    grid = (pts[..., :2] / pts[..., 2:3].clamp(min=1e-6)).view(B, OUT, OUT, 2)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def _gray(x):
    return (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1)


def _hue_matrix(h):
    """Rotation of the chroma plane in YIQ by angle 2*pi*h, as a per-sample RGB 3x3 matrix."""
    T = torch.tensor([[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]], device=h.device)
    Tinv = torch.linalg.inv(T)
    c, s = torch.cos(2 * math.pi * h), torch.sin(2 * math.pi * h)
    z, o = torch.zeros_like(c), torch.ones_like(c)
    R = torch.stack([torch.stack([o, z, z], -1), torch.stack([z, c, -s], -1), torch.stack([z, s, c], -1)], -2)
    return Tinv @ R @ T


def _color_jitter(x, B):
    device = x.device
    bright = torch.empty(B, 1, 1, 1, device=device).uniform_(0.7, 1.3)
    contrast = torch.empty(B, 1, 1, 1, device=device).uniform_(0.7, 1.3)
    sat = torch.empty(B, 1, 1, 1, device=device).uniform_(0.7, 1.3)
    hue = torch.empty(B, device=device).uniform_(-0.1, 0.1)
    x = (x * bright).clamp_(0, 1)
    mean = _gray(x).mean(dim=(1, 2, 3), keepdim=True)
    x = ((x - mean) * contrast + mean).clamp_(0, 1)
    g = _gray(x)
    x = ((x - g) * sat + g).clamp_(0, 1)
    x = torch.einsum("bij,bjhw->bihw", _hue_matrix(hue), x).clamp_(0, 1)
    return x


def _ta_color(x, ta_op, ta_mag, B):
    """Colour ops of TrivialAugmentWide: 6 Brightness, 7 Color, 8 Contrast, 9 Sharpness,
    10 Posterize, 11 Solarize, 12 AutoContrast, 13 Equalize (-> AutoContrast)."""
    device = x.device
    sign = torch.where(torch.rand(B, device=device) < 0.5, -1.0, 1.0)
    factor = (1.0 + 0.99 * ta_mag * sign).view(B, 1, 1, 1)
    op = ta_op.view(B, 1, 1, 1)

    out = x
    out = torch.where(op == 6, (x * factor).clamp(0, 1), out)
    g = _gray(x)
    out = torch.where(op == 7, ((x - g) * factor + g).clamp(0, 1), out)
    mean = g.mean(dim=(1, 2, 3), keepdim=True)
    out = torch.where(op == 8, ((x - mean) * factor + mean).clamp(0, 1), out)
    kernel = torch.tensor([[1, 1, 1], [1, 5, 1], [1, 1, 1]], device=device, dtype=x.dtype) / 13.0
    blurred = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), kernel.view(1, 1, 3, 3).expand(3, 1, 3, 3), groups=3)
    out = torch.where(op == 9, (blurred + (x - blurred) * factor).clamp(0, 1), out)
    bits = (8 - (ta_mag * 6).round()).clamp(2, 8).view(B, 1, 1, 1)
    levels = torch.pow(2.0, 8 - bits)
    out = torch.where(op == 10, torch.floor(x * 255 / levels) * levels / 255, out)
    thr = (1.0 - ta_mag).view(B, 1, 1, 1)
    out = torch.where(op == 11, torch.where(x >= thr, 1 - x, x), out)
    lo = x.amin(dim=(2, 3), keepdim=True)
    hi = x.amax(dim=(2, 3), keepdim=True)
    auto = ((x - lo) / (hi - lo).clamp(min=1e-3)).clamp(0, 1)
    out = torch.where((op == 12) | (op == 13), auto, out)
    return out


def _random_erasing(x, B):
    """RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0) per image."""
    device = x.device
    area = torch.empty(B, device=device).uniform_(0.02, 0.1) * OUT * OUT
    log_ratio = torch.empty(B, device=device).uniform_(math.log(0.3), math.log(3.3))
    ratio = torch.exp(log_ratio)
    h = torch.sqrt(area * ratio).clamp(1, OUT - 1)
    w = torch.sqrt(area / ratio).clamp(1, OUT - 1)
    top = torch.rand(B, device=device) * (OUT - h)
    left = torch.rand(B, device=device) * (OUT - w)
    ys = torch.arange(OUT, device=device).view(1, OUT, 1)
    xs = torch.arange(OUT, device=device).view(1, 1, OUT)
    mask = (ys >= top.view(B, 1, 1)) & (ys < (top + h).view(B, 1, 1)) & (xs >= left.view(B, 1, 1)) & (xs < (left + w).view(B, 1, 1))
    mask &= (torch.rand(B, device=device) < 0.5).view(B, 1, 1)
    return x.masked_fill(mask.unsqueeze(1), 0.0)


@torch.no_grad()
def gpu_augment(cfg, x_uint8):
    """``cfg.aug_tf`` for a uint8 batch [B, 3, S, S] on the GPU -> normalised float [B, 3, 224, 224]."""
    B, device = x_uint8.size(0), x_uint8.device
    ta_op = torch.randint(0, 14, (B,), device=device)       # TrivialAugmentWide: one op per image
    ta_mag = torch.randint(0, 31, (B,), device=device).float() / 30.0
    x = x_uint8.float().div_(255.0)
    x = _warp(x, _sample_geometry(B, device, ta_op, ta_mag))
    x = _color_jitter(x, B)
    x = _ta_color(x, ta_op, ta_mag, B)
    x = _normalize(cfg, x)
    return _random_erasing(x, B)


# ----------------------------------------------------------------------------------------
# Split objects
# ----------------------------------------------------------------------------------------
class GpuSplit:
    """One split held on the GPU. Iterating yields (images, labels) batches like a DataLoader;
    ``train=True`` shuffles and augments, otherwise the evaluation transform is applied.
    Exposes ``samples``/``dataset`` so YNLT can still find the source files."""

    def __init__(self, cfg, bank, records, batch_size, train=False, seed=0):
        self.cfg, self.bank, self.batch_size, self.train = cfg, bank, int(batch_size), train
        self.records = list(records)
        self.samples = [(r["_abs_path"], int(r["label_idx"])) for r in self.records]
        self.rows = bank.rows([r["_abs_path"] for r in self.records])
        self.labels = torch.tensor([int(r["label_idx"]) for r in self.records], dtype=torch.long, device=bank.device)
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.dataset = self
        self.classes = list(cfg.CANONICAL_CLASSNAMES_LIST)
        self.targets = [label for _, label in self.samples]

    def __len__(self):
        return math.ceil(len(self.records) / self.batch_size)

    def __iter__(self):
        n = len(self.records)
        order = torch.randperm(n, generator=self.generator) if self.train else torch.arange(n)
        order = order.to(self.bank.device)
        for start in range(0, n, self.batch_size):
            idx = order[start:start + self.batch_size]
            raw = self.bank.images[self.rows[idx]]
            images = gpu_augment(self.cfg, raw) if self.train else gpu_eval_transform(self.cfg, raw)
            yield images, self.labels[idx].cpu()

    @torch.no_grad()
    def eval_tensors(self):
        """All images under the evaluation transform, as one tensor (for feature caching)."""
        return torch.cat([images for images, _ in self], dim=0), self.labels


def get_gpu_splits(cfg, split_csv, device, batch_size=None, train_fraction=None, seed=0, test_csv=None, verbose=True):
    """GPU counterpart of ``data.get_split_loaders`` with the same keys and leak check."""
    batch_size = batch_size or cfg.batch_size
    records = read_split_records(split_csv)
    test_records = read_split_records(test_csv) if test_csv and test_csv != split_csv else records
    for rec in records + (test_records if test_records is not records else []):
        rec["_abs_path"] = rec["path"] if os.path.isabs(rec["path"]) else os.path.normpath(os.path.join(cfg.REPO_ROOT, rec["path"]))

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

    needed = sorted({r["_abs_path"] for r in train_records + val_records + sum(tests.values(), [])})
    bank = ImageBank(cfg, needed, device, verbose=verbose)

    splits = {
        "train": GpuSplit(cfg, bank, train_records, batch_size, train=True, seed=seed),
        "val": GpuSplit(cfg, bank, val_records, batch_size) if val_records else None,
    }
    for name, recs in tests.items():
        splits[name] = GpuSplit(cfg, bank, recs, batch_size) if recs else None
    splits["sizes"] = {"train": len(train_records), "val": len(val_records), **{n: len(r) for n, r in tests.items()}}
    splits["bank"] = bank
    if verbose:
        print(
            "Split  " + "  ".join(f"{k}={v}" for k, v in splits["sizes"].items())
            + (f"  (train fraction {train_fraction})" if train_fraction and train_fraction < 1.0 else "")
        )
    return splits
