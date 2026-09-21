"""Project settings. Point LAB_DATA_DIR / OOD_DATA_DIR at your image folders if needed."""

import os

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
# --------------------------------------------------------------------------------------
batch_size = 32
num_workers = 4
pin_memory = True
use_amp = True

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
    v2.ToTensor(),
    v2.Normalize(mean=normalization_parameters["mean"], std=normalization_parameters["std"]),
    v2.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
])

basic_tf = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToTensor(),
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
    # ConvNeXt-V1 counterpart of the V2 checkpoint above: IN-22k pretraining, IN-1k fine-tune.
    "convnext": "convnext_tiny.fb_in22k_ft_in1k",
    "resnet50": "resnet50.a1_in1k",
    "efficientnet": "tf_efficientnetv2_b0.in1k",
    "vit": "vit_base_patch16_224.augreg_in21k_ft_in1k",
}

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
#   "original"  - re-open the source file and patch the full-resolution image (paper text)
#   "input224"  - patch the 224x224 network input (what the archive notebooks did for Table 1)
ynlt_patch_source = "original"
ynlt_patch_batch_size = 16
