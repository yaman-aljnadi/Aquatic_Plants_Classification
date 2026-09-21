import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import config as cfg

_PATCH_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=cfg.normalization_parameters["mean"], std=cfg.normalization_parameters["std"]),
])


class ImagePatchedDataset(Dataset):
    def __init__(self, image, label=None, transform=_PATCH_TF):
        self.image = image
        self.label = label
        self.transform = transform
        self.patches = []
        if isinstance(self.image, Image.Image):
            if self.image.mode != "RGB":
                self.image = self.image.convert("RGB")
            self.image = np.array(self.image)
        elif isinstance(self.image, np.ndarray):
            if self.image.ndim == 2 or (self.image.ndim == 3 and self.image.shape[2] == 1):
                self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)

    def pad_image_for_patch_extraction(self, patch_size, stride):
        h, w, _ = self.image.shape
        pad_h = (stride - (h - patch_size) % stride) % stride if h > patch_size else 0
        pad_w = (stride - (w - patch_size) % stride) % stride if w > patch_size else 0
        if pad_h > 0 or pad_w > 0:
            border = cv2.BORDER_REFLECT if cfg.padding_logic == "reflect" else cv2.BORDER_CONSTANT
            kwargs = {} if cfg.padding_logic == "reflect" else {"value": [255, 255, 255]}
            self.image = cv2.copyMakeBorder(self.image, 0, int(pad_h), 0, int(pad_w), border, **kwargs)

    def extract_patches_by_size(self, patch_size=224, stride=None):
        """Append this scale's patches to self.patches and return just the new ones.

        Patches are views into self.image, so multi-scale extraction stays cheap.
        Returning only the current scale matters: self.patches holds several sizes
        at once, which cannot be stacked into one array.
        """
        if stride is None:
            stride = patch_size
        self.pad_image_for_patch_extraction(patch_size, stride)
        h, w, _ = self.image.shape
        new_patches = []
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                patch = self.image[y:y + patch_size, x:x + patch_size]
                if np.all(patch == patch[0, 0]):
                    continue
                new_patches.append(patch)
        self.patches.extend(new_patches)
        return new_patches

    def extract_patches_multi_scale(self, scales=None, overlap=0.0):
        scales = scales if scales is not None else [0.2, 0.3, 0.4, 0.5]
        if isinstance(overlap, float):
            overlap = [overlap] * len(scales)
        h, w, _ = self.image.shape
        min_dim = min(h, w)
        for scale, ov in zip(scales, overlap):
            patch_size = int(min_dim * scale)
            stride = max(int(patch_size * (1 - ov)), 1)
            self.extract_patches_by_size(patch_size=patch_size, stride=stride)

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch = self.patches[idx]
        if isinstance(patch, np.ndarray):
            patch = Image.fromarray(patch)
        if self.transform:
            patch = self.transform(patch)
        label = torch.tensor(-1 if self.label is None else self.label, dtype=torch.long)
        return patch, label


def get_dataloader(image, label=None, batch_size=1, shuffle=False):
    dataset = ImagePatchedDataset(image=image, label=label)
    if cfg.patching_method == "by-size":
        dataset.extract_patches_by_size(patch_size=cfg.bs_patch_size, stride=cfg.bs_stride)
    else:
        dataset.extract_patches_multi_scale(scales=cfg.ms_scale, overlap=cfg.ms_overlap)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
