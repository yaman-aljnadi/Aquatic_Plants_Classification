"""Project settings. Point LAB_DATA_DIR / OOD_DATA_DIR at your image folders if needed."""

import os

# Pure noise on Windows, printed by every dataloader worker on first model download.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
# numpy's OpenBLAS allocates a thread-buffer set per process; with eight DataLoader
# workers that has failed with "Memory allocation still failed" and hung training.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "runs")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def _first_dir(*paths):
    for path in paths:
        if path and os.path.isdir(path):
            return path
    return None


LAB_DATA_DIR = _first_dir(
    os.environ.get("AQUATIC_LAB_DATA"),
    os.path.join(REPO_ROOT, "Aquatic Plant lab"),
)
OOD_DATA_DIR = _first_dir(
    os.environ.get("AQUATIC_OOD_DATA"),
    os.path.join(REPO_ROOT, "Aquatic Plant outdoor"),
)
# Web-sourced field photographs (GBIF-mediated iNaturalist observations), one folder per
# species with underscores instead of spaces. Several photos can belong to one observation,
# so splits are grouped by ``gbif_occurrence_key`` from the download manifest.
WEB_DATA_DIR = _first_dir(
    os.environ.get("AQUATIC_WEB_DATA"),
    os.path.join(REPO_ROOT, "Larger_Plants_Dataset", "aquatic_dataset_report", "aquatic_plant_images"),
)
WEB_MANIFEST = os.path.join(WEB_DATA_DIR, "download_manifest.csv") if WEB_DATA_DIR else None

# Frozen split lists written by scripts/build_splits.py and read by train.py / evaluate.py.
SPLITS_DIR = os.path.join(PROJECT_ROOT, "data", "splits")
DATA_CONFIGS = ("lab", "web", "lab+web", "lab+web+field")


def split_csv_path(data_config, seed):
    """data/splits/<data>_seed<N>.csv ('lab+web' is written as 'lab_web')."""
    return os.path.join(SPLITS_DIR, f"{data_config.replace('+', '_')}_seed{seed}.csv")


main_data_dir = LAB_DATA_DIR
hand_test_data_dir = OOD_DATA_DIR

# --------------------------------------------------------------------------------------
# Classes
# --------------------------------------------------------------------------------------
CANONICAL_CLASSNAMES_LIST = [
    "Brasenia schreberi",
    "Cabomba",
    "Ceratophyllum demersum",
    "Elodea canadensis",
    "Heteranthera dubia",
    "Hydrocharis morsus-ranae",
    "Myriophyllum sibiricum",
    "Myriophyllum spicatum",
    "Najas flexilis",
    "Nitellopsis obtusa",
    "Nuphar variegata",
    "Nymphaea odorata",
    "Potamogeton crispus",
    "Potamogeton gramineus",
    "Potamogeton illinoensis",
    "Potamogeton natans",
    "Potamogeton praelongus",
    "Potamogeton richardsonii",
    "Potamogeton robbinsii",
    "Ranunculus aquatilis",
    "Vallisneria americana",
]
CANONICAL_CLASS_TO_INDEX = {name: i for i, name in enumerate(CANONICAL_CLASSNAMES_LIST)}
CANONICAL_INDEX_TO_CLASS = {i: name for i, name in enumerate(CANONICAL_CLASSNAMES_LIST)}
CANONICAL_NUM_CLASSES = len(CANONICAL_CLASSNAMES_LIST)

INVASIVE_SPECIES_NAMES = [
    "Hydrocharis morsus-ranae",
    "Myriophyllum spicatum",
    "Nitellopsis obtusa",
    "Potamogeton crispus",
    "Cabomba",
]
INVASIVE_INDICES = [CANONICAL_CLASS_TO_INDEX[name] for name in INVASIVE_SPECIES_NAMES]

# --------------------------------------------------------------------------------------
# Hardware defaults for RTX 3070 Ti (8 GB) + i9-12900K
# Paper training used batch_size=4 on a 4 GB A500. Frozen 224x224 heads fit 32 easily here.
# persistent_workers matters on Windows: without it every loader respawns its workers
# each epoch (3 loaders x 250 epochs), which made a 121-image run take two hours.
# --------------------------------------------------------------------------------------
batch_size = 32
num_workers = 8          # CPU loader path and one-off image-bank decoding
eval_workers = 2         # validation / test loaders (plain resize only), CPU path
pin_memory = True
persistent_workers = True
prefetch_factor = 4
use_amp = True

# GPU-resident data path (src/gpu_data.py): every image is decoded once, squashed to
# gpu_cache_size x gpu_cache_size, cached in CACHE_DIR and kept on the GPU as uint8; the
# training augmentation runs as batched GPU kernels. train.py uses it by default on CUDA
# (--loader cpu restores the DataLoader path).
gpu_cache_size = 320
CACHE_DIR = os.path.join(PROJECT_ROOT, "data", "cache")

# --------------------------------------------------------------------------------------
# Augmentation / splits
# --------------------------------------------------------------------------------------
normalization_parameters = {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}

aug_tf = v2.Compose([
    v2.RandomResizedCrop(size=(224, 224), scale=(0.1, 1.0)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.5),
    v2.RandomRotation(degrees=90),
    v2.RandomPerspective(distortion_scale=0.15, p=0.5, interpolation=InterpolationMode.BICUBIC),
    v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    v2.TrivialAugmentWide(interpolation=InterpolationMode.BICUBIC),
    # v2.ToTensor() is deprecated; this pair is its documented replacement and is
    # numerically equivalent (uint8 HWC -> float32 CHW scaled to [0, 1]).
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=normalization_parameters["mean"], std=normalization_parameters["std"]),
    v2.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
])

basic_tf = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=normalization_parameters["mean"], std=normalization_parameters["std"]),
])

train_percent = 0.6
val_percent = 0.2
test_percent = 0.2
random_seed = 8

# --------------------------------------------------------------------------------------
# Model / training (paper recipe, larger batch for this GPU)
# --------------------------------------------------------------------------------------
BACKBONE_PRESETS = {
    "convnextv2": "convnextv2_tiny.fcmae_ft_in22k_in1k",
    # Smaller ConvNeXt-V2 variants: candidates for the underwater-rover compute budget.
    # Nano has an IN-22k checkpoint; Pico and Atto are only released with IN-1k fine-tuning.
    "convnextv2_nano": "convnextv2_nano.fcmae_ft_in22k_in1k",
    "convnextv2_pico": "convnextv2_pico.fcmae_ft_in1k",
    "convnextv2_atto": "convnextv2_atto.fcmae_ft_in1k",
    # ConvNeXt-V1 counterpart of the V2 checkpoint above: IN-22k pretraining, IN-1k fine-tune.
    "convnext": "convnext_tiny.fb_in22k_ft_in1k",
    "resnet50": "resnet50.a1_in1k",
    "efficientnet": "tf_efficientnetv2_b0.in1k",
    "vit": "vit_base_patch16_224.augreg_in21k_ft_in1k",
}
SMALL_CONVNEXTV2_PRESETS = ("convnextv2_nano", "convnextv2_pico", "convnextv2_atto")

train_timm_model_name = BACKBONE_PRESETS["convnextv2"]
classification_head_type = "gated_attention"
classifier_head_dropout = 0.1
gated_attention_dim = 168
mlp_hidden_dim = 42

num_epochs = 250
label_smoothing = 0.05
weight_decay = 2e-5
lr_initial = 0.0005
lr_cosine_minimum = 0.00002

# --------------------------------------------------------------------------------------
# YNLT
# --------------------------------------------------------------------------------------
do_critical_review = True
critical_confusion_level = 1
ignore_invasive_predictions = True
margin_weight = 1
entropy_weight = 1
confidence_weight = 1
margin_threshold = 0.2
entropy_threshold = 2.0
confidence_threshold = 0.5

padding_logic = "reflect"
patching_method = "multi-scale"
ms_scale = [0.2, 0.3, 0.4, 0.5]
ms_overlap = [0.0, 0.1, 0.2, 0.3]
bs_patch_size = 224
bs_stride = None

# Where the second look cuts its patches from.
#   "original"  - re-open the source file and patch the full-resolution image
#   "input224"  - patch the 224x224 network input (what the archive notebooks did for Table 1)
# On the 203/42 collection the two sources agreed within one accuracy point on every
# column while "original" cost 5-10x more per reviewed image, so input224 is the default.
ynlt_patch_source = "input224"
ynlt_patch_batch_size = 16
