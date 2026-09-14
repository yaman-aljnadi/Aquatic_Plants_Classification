# cnn_utils/patching.py

import cv2
import numpy as np
from PIL import Image

import a1_cnn_ynlt.config as cfg

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

normalization_parameters = {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225]
}

basic_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=normalization_parameters["mean"], std=normalization_parameters["std"])
])
class ImagePatchedDataset(Dataset):
    """Takes a single input image and returns a dataset of patches of the image.
    Args:
        image (np.ndarray) or PIL image: Input image to be patched.
        label (int, optional): Label for the image. Defaults to None.
        
        patching_method (str): one of the following:
            - 'by-size': Extract patches of a specified size and stride.
            - 'multi-scale': Extract patches at multiple scales.
    """
    def __init__(self, image, label = None, transform=basic_tf):
        self.image = image
        self.label = label
        self.transform = transform
        self.patches = []

        # Make sure the image is in the correct format - RGB and numpy array
        if isinstance(self.image, Image.Image):
            if self.image.mode != 'RGB':
                self.image = self.image.convert('RGB')
            self.image = np.array(self.image)
        
        elif isinstance(self.image, np.ndarray):
            if self.image.ndim == 2 or (self.image.ndim == 3 and self.image.shape[2] == 1):
                self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
        
        # print("Run one of the patch extraction methods to extract patches from the image.")
        
    def pad_image_for_patch_extraction(self, patch_size, stride):
        """Pad the image to ensure that patches can be extracted without going out of bounds.
        """
        h, w, _ = self.image.shape
        pad_h = (stride - (h - patch_size) % stride) % stride if h > patch_size else 0
        pad_w = (stride - (w - patch_size) % stride) % stride if w > patch_size else 0
        
        if pad_h > 0 or pad_w > 0:
            if cfg.padding_logic == 'reflect':
                self.image = cv2.copyMakeBorder(self.image, 0, int(pad_h), 0, int(pad_w), cv2.BORDER_REFLECT)
            else:
                self.image = cv2.copyMakeBorder(self.image, 0, int(pad_h), 0, int(pad_w), cv2.BORDER_CONSTANT, value=[255, 255, 255])
    
    def extract_patches_by_size(self, patch_size = 224, stride = None):
        """Extract patches from the image."""
        self.pad_image_for_patch_extraction(patch_size, stride)
        if stride is None:
            stride = patch_size
            
        patches = []
        h, w, _ = self.image.shape
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                patch = self.image[y:y + patch_size, x:x + patch_size]
                if np.all(patch == patch[0, 0]):
                    continue
                
                patches.append(patch)
                
        self.patches.extend(patches)
        
        return np.array(patches)

    def extract_patches_multi_scale(self, scales=[0.2, 0.3, 0.4, 0.5], overlap=0.0):
        """
        Extract patches at multiple scales from the image.
        Uses the extract_patches_by_size method to extract patches at each scale.
        Sets stride according to the overlap parameter.
        
        Args:
            scales (list): List of scales to extract patches at.
            overlap (float): Overlap between patches 
                - if a single float is given, it is used for all scales.
                - if a list is given, it should have the same length as scales.
        """
        if isinstance(overlap, float):
            overlap = [overlap] * len(scales)
        
        if len(overlap) != len(scales):
            raise ValueError("If overlap is a list, it must have the same length as scales.")
        
        # image dims
        h, w, _ = self.image.shape
        min_dim = min(h, w)
        
        for scale, overlap in zip(scales, overlap):
            patch_size = int(min_dim * scale)
            stride = int(patch_size * (1 - overlap))
            
            # Extract patches at this scale
            self.extract_patches_by_size(patch_size=patch_size, stride=stride)       
            
    def clear_patches(self):
        self.patches = []
        
    def __len__(self):
        """Return the number of patches."""
        return len(self.patches)

    def __getitem__(self, idx):
        """Return a patch and its corresponding label."""
        patch = self.patches[idx]
        label = self.label
        
        if isinstance(patch, np.ndarray):
            patch = Image.fromarray(patch)
        if self.transform:
            patch = self.transform(patch)
        if label is not None:
            label = torch.tensor(label, dtype=torch.long)  
        else:
            label = torch.tensor(-1, dtype=torch.long)      
        
        return patch, label

def get_dataloader_by_size(image, label=None, patch_size=224, stride=None, batch_size=1, shuffle=False):
    """Get a DataLoader for the image patches extracted by size."""
    dataset = ImagePatchedDataset(image=image, label=label)
    dataset.extract_patches_by_size(patch_size=patch_size, stride=stride)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return dataloader

def get_dataloader_multi_scale(image, label=None, scales=[0.2, 0.3, 0.4, 0.5], overlap=0.0, batch_size=1, shuffle=False):
    """Get a DataLoader for the image patches extracted at multiple scales."""
    dataset = ImagePatchedDataset(image=image, label=label)
    dataset.extract_patches_multi_scale(scales=scales, overlap=overlap)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return dataloader

def get_dataloader(image, label=None, batch_size=1, shuffle=False):
    """Get a DataLoader for the image patches..."""
    if cfg.patching_method == 'by-size':
        return get_dataloader_by_size(
            image, label=label, patch_size=cfg.patch_size, stride=cfg.stride,
            batch_size=batch_size, shuffle=shuffle
        )
    elif cfg.patching_method == 'multi-scale':
        return get_dataloader_multi_scale(
            image, label=label, scales=cfg.ms_scale, overlap=cfg.ms_overlap,
            batch_size=batch_size, shuffle=shuffle
        )
    else:
        raise ValueError("Invalid patching method. Choose 'by-size' or 'multi-scale'.")
        
