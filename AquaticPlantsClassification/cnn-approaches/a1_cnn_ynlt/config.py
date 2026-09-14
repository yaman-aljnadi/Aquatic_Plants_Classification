# config.py

import os
import shutil
import importlib.util

from torchvision import transforms
from torchvision.transforms import v2, InterpolationMode

# ===============================================================================================
# Canonical Data Paths and Configurations
# ===============================================================================================
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
_REPO_ROOT = os.path.abspath(os.path.join(_CODE_ROOT, ".."))

def _first_existing(*paths):
    for path in paths:
        if path and os.path.isdir(path):
            return path
    return next((p for p in paths if p), None)

PROJECT_ROOT = _first_existing(
    os.environ.get("AQUATIC_PROJECT_ROOT"),
    "/home/takayuki/Desktop/summer2025/plants/plant_classification",
    _CODE_ROOT,
)
THIS_FILE_PATH = os.path.abspath(__file__)

saved_configs_path = os.path.join(_REPO_ROOT, "saved_configs")
os.makedirs(saved_configs_path, exist_ok=True)

DATA_DIR = _first_existing(
    os.environ.get("AQUATIC_LAB_DATA"),
    os.path.join(PROJECT_ROOT, "../data/AquaticPlantLabData"),
    os.path.join(_REPO_ROOT, "Aquatic Plant lab"),
)

PREPROCESSED_DIR = os.path.join(DATA_DIR, "preprocessed") if DATA_DIR else None
SQUARED_DIR = os.path.join(DATA_DIR, "squared") if DATA_DIR else None
SQUARED_RELECTED_DIR = os.path.join(DATA_DIR, "squared_reflected") if DATA_DIR else None
HANDS_DIR = _first_existing(
    os.environ.get("AQUATIC_OOD_DATA"),
    os.path.join(DATA_DIR, "hand") if DATA_DIR else None,
    os.path.join(_REPO_ROOT, "Aquatic Plant outdoor"),
)

_DEFAULT_CLASSNAMES = [
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

_class_names_file = os.path.join(DATA_DIR, "class_names.txt") if DATA_DIR else None
if _class_names_file and os.path.isfile(_class_names_file):
    with open(_class_names_file, "r", encoding="utf-8") as f:
        CANONICAL_CLASSNAMES_LIST = sorted([line.strip() for line in f if line.strip()])
else:
    CANONICAL_CLASSNAMES_LIST = list(_DEFAULT_CLASSNAMES)

CANONICAL_CLASS_TO_INDEX = {name: i for i, name in enumerate(CANONICAL_CLASSNAMES_LIST)}
CANONICAL_INDEX_TO_CLASS = {i: name for i, name in enumerate(CANONICAL_CLASSNAMES_LIST)}
CANONICAL_NUM_CLASSES = len(CANONICAL_CLASS_TO_INDEX)

INVASIVE_SPECIES_NAMES = [ 
    "Hydrocharis morsus-ranae",
    "Myriophyllum spicatum",
    "Nitellopsis obtusa",
    "Potamogeton crispus",
    "Cabomba" 
]
INVASIVE_INDICES = [CANONICAL_CLASS_TO_INDEX[name] for name in INVASIVE_SPECIES_NAMES]

# ===============================================================================================
# Data Preprocessing and Augmentation 
# ==================================================== TODO: Implement these and train

squaring_logic = 'white'  # Options: 'white' or 'reflect'  

normalization_parameters = {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225]
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

# ===================================================================================================
# Models and Training Parameters
# ==================================================== # TODO: Later

# phase 6 - dataset is reconfigured, attention_dim=168

# Paths and Directories ------------------------------------------------------------------
TRAINING_BASE_DIR = os.path.join(_REPO_ROOT, "training", "phase7") 
checkpoint_path = os.path.join(TRAINING_BASE_DIR, "checkpoints")
models_path = os.path.join(TRAINING_BASE_DIR, "models")
base_logs_path = os.path.join(TRAINING_BASE_DIR, "runs/convnextv2")

os.makedirs(checkpoint_path, exist_ok=True)
os.makedirs(models_path, exist_ok=True)
os.makedirs(base_logs_path, exist_ok=True)

main_data_dir = _first_existing(SQUARED_RELECTED_DIR, SQUARED_DIR, DATA_DIR)

train_percent = 0.6
val_percent = 0.2
test_percent = 0.2
random_seed = 8


# Timm model architecture names -----------------------------------------------------------
train_timm_model_names = [
    "convnextv2_tiny.fcmae_ft_in1k",           # 0 - ConvNeXtV2 Tiny (28M params)
    "convnextv2_tiny.fcmae_ft_in22k_in1k",     # 1 - ConvNeXtV2 Tiny (28M params, IN22k finetuned)
    "convnextv2_nano.fcmae_ft_in1k",           # 2 - ConvNeXtV2 Nano (15M params)
    "convnextv2_nano.fcmae_ft_in22k_in1k",     # 3 - ConvNeXtV2 Nano (15M params, IN22k finetuned)
    "convnextv2_pico.fcmae_ft_in1k",           # 4 - ConvNeXtV2 Pico (9M params)
    "convnextv2_femto.fcmae_ft_in1k",          # 5 - ConvNeXtV2 Femto (5M params)
    "convnextv2_atto.fcmae_ft_in1k",           # 6 - ConvNeXtV2 Atto (3M params), Smallest
    "tf_efficientnetv2_b0.in1k",               # 7 - EfficientNetV2-B0
    "swinv2_cr_tiny_ns_224.sw_in1k",           # 8 - SwinV2 Tiny
    "focalnet_tiny_srf.ms_in1k",               # 9 - FocalNet Tiny
    "convnextv2_base.fcmae_ft_in22k_in1k"      # 10 - ConvNeXtV2 Base (88M params)
]

train_timm_model_name = train_timm_model_names[1]

classification_head_type = 'gated_attention' # Options: 'linear', 'gated_attention', 'mlp'
classifier_head_dropout = 0.1

gated_attention_dim = 168   # If None, it will be set to input_dim // 4

mlp_hidden_dim = 42

# Training run parameters ---------------------------------------------------

training_run_name = "ph7_gated_attn_actual"
num_epochs = 250

batch_size = 4

label_smoothing = 0.05  # This will teach the model to not be overconfident

weight_decay = 2e-5

lr_scheme = 'cosine'  # Options: 'cosine', 'fixed', 'step' 
lr_initial = 0.0005

lr_cosine_minimum = 0.00002
lr_cosine_tmax_epochs = num_epochs

# lr_step_size = 10
# lr_step_gamma = 0.1

# checkpoint_interval = 100
validation_interval = 1

ood = True
# Early stopping parameters ---------------------------------------------------
# We stop when both loss and accuracy do not improve for a certain number of epochs, respectively
early_stopping = False
patience_loss = 30        
patience_accuracy = 30  


# ===================================================================================================
# Evaluation Parameters
# ===================================================================================================

hand_test_data_dir = HANDS_DIR

# Models to evaluate on -----------------------------------------------------------------------------
# These models are used in the comparison section of the report generated

# Model A is obtained from the get_model function as per the above configs
# Model B can be specified below - these are used by get_model_b function in model.py
# model_b_timm_name = train_timm_model_names[1] 
# model_b_classification_head_type = 'linear'  # Options: 'linear', 'gated_attention', 'mlp'
# model_b_gated_attention_dim = 42  

#exp19 - good
p54b_gated_attn = "/home/takayuki/Desktop/summer2025/plants/training/phase5/models/best_model_07-10_18-28--12_p54b_gated_attn_convnextv2_tiny.fcmae_ft_in22k_in1k.pth"
p54b_linear = "/home/takayuki/Desktop/summer2025/plants/training/phase5/models/best_model_07-10_19-57--50_p54b_linear_convnextv2_tiny.fcmae_ft_in22k_in1k.pth"

ph6_gated_attn_best = "/home/takayuki/Desktop/summer2025/plants/training/phase6/models/best_model_07-14_14-16--00_p6_gated_attn_convnextv2_tiny.fcmae_ft_in22k_in1k.pth"
ph6_linear_best = "/home/takayuki/Desktop/summer2025/plants/training/phase6/models/best_model_07-14_14-56--10_p6_linear_convnextv2_tiny.fcmae_ft_in22k_in1k.pth"
ph6_gated_final = "/home/takayuki/Desktop/summer2025/plants/training/phase6/models/final_model_07-14_14-16--00_p6_gated_attn_convnextv2_tiny.fcmae_ft_in22k_in1k.pth"
ph6_linear_final = "/home/takayuki/Desktop/summer2025/plants/training/phase6/models/final_model_07-14_14-56--10_p6_linear_convnextv2_tiny.fcmae_ft_in22k_in1k.pth"

ph62_gattn_best_acc = "/home/takayuki/Desktop/summer2025/plants/training/phase6/models/best_acc_model_07-15_11-49--31_ph62_gated_attn_convnextv2_tiny.fcmae_ft_in22k_in1k.pth"
ph62_gattn_best_loss = "/home/takayuki/Desktop/summer2025/plants/training/phase6/models/best_loss_model_07-15_11-49--31_ph62_gated_attn_convnextv2_tiny.fcmae_ft_in22k_in1k.pth"

ph7_gated_attn = "/home/takayuki/Desktop/summer2025/plants/training/phase7/models/final_model_07-31_04-19--55_ph7_gated_attn_actual_convnextv2_tiny.fcmae_ft_in22k_in1k.pth"
ph7_gated_attn_best_ood_loss = "/home/takayuki/Desktop/summer2025/plants/training/phase7/models/best_ood_loss_model_07-31_04-19--55_ph7_gated_attn_actual_convnextv2_tiny.fcmae_ft_in22k_in1k.pth"
ph7_gated_attn_best_ood_acc = "/home/takayuki/Desktop/summer2025/plants/training/phase7/models/best_ood_acc_model_07-31_04-19--55_ph7_gated_attn_actual_convnextv2_tiny.fcmae_ft_in22k_in1k.pth"

best_model_path = ph7_gated_attn_best_ood_acc

# best_model_b_path = phase5_first_champ  # Model B for A/B testing

# Patching ------------------------------------------------------------------------------------------
padding_logic = 'reflect'        # Options: 'white' or 'reflect'

patching_method = 'multi-scale'  # Options: 'by-size' or 'multi-scale'

## multi-scale
ms_scale = [0.2, 0.3, 0.4, 0.5]
ms_overlap = [0.0, 0.1, 0.2, 0.3]  

## by-size
bs_patch_size = 224
bs_stride = None       # If None, it will be set to patch_size

# Critical Review ------------------------------------------------------------------------------------
do_critical_review = True          # If True, will review critical predictions
critical_confusion_level = 1       # Threshold for critical confusion
ignore_invasive_predictions = True # If the prectioin is invasive - don't review it?

## Confusion Level Prameters
margin_weight = 1
entropy_weight = 1
confidence_weight = 1   
        
margin_threshold = 0.2            # If the margin is below this, it is considered confused
entropy_threshold = 2.0           # If the entropy is above this, it is considered confused
confidence_threshold = 0.5        # If the confidence is below this, it is considered confused

# Report generation -----------------------------------------------------------

expriment_name = "ph7_gated_attn_best_ood_loss" 

EXPERIMENTS_DIR = os.path.join(_REPO_ROOT, "experiments")
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
experiment_number = 1 + len(os.listdir(EXPERIMENTS_DIR))
experiment_name = f"exp-{experiment_number}_{expriment_name}"

A4 = (8.27, 11.69)  
LETTER = (8.5, 11)
PDF_PAGE_SIZE = (15, 12) # (w, h) Options: A4, LETTER, (custom size)  # Also used for confusion matrix dimensions

report_config_groups = {
    "CRITICAL REVIEW": ['do_critical_review', 'critical_confusion_level', 'ignore_invasive_predictions'],
    "CONFUSION METRICS": ['margin_threshold', 'entropy_threshold', 'confidence_threshold'],
    "CONFUSION_METRICS WEIGHTS": ['margin_weight', 'entropy_weight', 'confidence_weight'],
    "PATCHING": ['padding_logic', 'patching_method', 'ms_scale', 'ms_overlap', 'bs_patch_size', 'bs_stride'],
}


# ==============================================================================
#                      PLOTTING & FIGURE GENERATION
# ==============================================================================
import matplotlib.pyplot as plt

FIGURE_SAVE_DIR = os.path.join(PROJECT_ROOT, "../figures26")

# --- Base Style ---
plt.style.use('seaborn-v0_8-ticks')

# --- Figure Width Calculations ---
FIG_WIDTH_SINGLE_COL_INCH = 3.5
# Double column width = 170 mm
FIG_WIDTH_DOUBLE_COL_INCH = 7

GOLDEN_RATIO = 1.618
# Calculate aesthetically pleasing heights
FIG_HEIGHT_SINGLE_COL_INCH = FIG_WIDTH_SINGLE_COL_INCH / GOLDEN_RATIO
FIG_HEIGHT_DOUBLE_COL_INCH = FIG_WIDTH_DOUBLE_COL_INCH / GOLDEN_RATIO

# --- Font Properties to Match Elsevier's 'Times' Style (8-12 pt range) ---
# Elsevier recommends 'Times' font family. 'Times New Roman' is the standard implementation.
FONT_FAMILY = "Times New Roman"

BASE_FONTSIZE = 9      # For main text like axis labels.
TITLE_FONTSIZE = 10    # Slightly larger for titles if shown.
TICK_LABEL_FONTSIZE = 8  # Ticks can be slightly smaller for clarity.
LEGEND_FONTSIZE = 8
ANNOTATION_FONTSIZE = 8
ANNOTATION_FONTSIZE_LARGE = 14

# --- Line and Marker Properties for Print ---
LINE_WIDTH = 1        # Ensures lines are clearly visible when printed.
MARKER_SIZE = 5         # Clear, but not overwhelming, marker size.

# --- Other Properties ---
FIGURE_DPI = 300           # Standard for rasterized elements in print.
VECTOR_FORMATS = ('.pdf', '.svg', '.eps')
AXIS_LABEL_PADDING = 5     # Space between axis and its label.
AXIS_GUIDELINE_COLOR = 'grey'
AXIS_GUIDELINE_WIDTH = 0.8
AXIS_LINE_WIDTH = 0.8

# --- Chart-Specific Properties ---
# For plots with variable numbers of items, like the classwise metrics bar chart.
BAR_CHART_HEIGHT_PER_ITEM_INCH = 0.15 # Vertical space allocated for each bar.
BAR_CHART_BASE_HEIGHT_INCH = 1.5     # Base height for axes, titles, etc.
BAR_CHART_THICKNESS = 0.8            # Thickness of the individual bars (0.0 to 1.0)
BAR_CHART_X_AXIS_PADDING = 0.2       # Padding for the x-axis limits (e.g., 1.0 -> 1.1)

# --- Manuscript Mode Toggles ---
SHOW_TITLES_IN_PLOTS = False 

# --- Color Definitions ---
COLOR_PALETTE = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3'] 
INVASIVE_HIGHLIGHT_COLOR = 'red'
NEUTRAL_COLOR = 'black'
# DIST_INVASIVE_COLOR = 'lightcoral'
# DIST_NON_INVASIVE_COLOR = 'lightblue'

DIST_INVASIVE_COLOR = 'lightcoral'  # Use the consistent red for invasive species
DIST_NON_INVASIVE_COLOR = COLOR_PALETTE[0]      # Use the primary blue for non-invasive

CMAP_BLUES = 'Blues'

PRIMARY_COLOR = COLOR_PALETTE[0]   # e.g., '#377eb8' for the 'good' metric
SECONDARY_COLOR = COLOR_PALETTE[1] # e.g., '#ff7f00' for the 'error' metric
    
# --- Chart-Specific Properties ---
# For the distribution chart of invasive vs non-invasive species.
DIST_CHART_FIGSIZE = (FIG_WIDTH_DOUBLE_COL_INCH, FIG_HEIGHT_SINGLE_COL_INCH * 1.8)
DIST_CHART_XTICK_ROTATION = 45
DIST_CHART_XTICK_ALIGNMENT = 'right' # Alignment for rotated labels
DIST_CHART_Y_AXIS_TOP_MARGIN_FACTOR = 1 # e.g., 1.15 gives 15% padding at the top
BAR_LABEL_PADDING_FACTOR = 0.01 # Padding for count labels above bars, as a factor of max count.
DIST_CHART_YLABEL_PADDING = 12 # Space between y-axis label and the axis itself
# --- Legend Properties ---
LEGEND_LABEL_INVASIVE = 'Invasive Species'
LEGEND_LABEL_NON_INVASIVE = 'Non-Invasive/Other'
DIST_CHART_LEGEND_LOCATION = 'upper right'
# --- Grid Properties ---
GRID_STYLE = '--'
GRID_ALPHA = 0.7

def apply_matplotlib_styles():
    """Applies the global Matplotlib settings"""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': FONT_FAMILY,
        'font.size': BASE_FONTSIZE,
        
        'axes.titlesize': TITLE_FONTSIZE,
        'axes.labelsize': BASE_FONTSIZE,
        'xtick.labelsize': TICK_LABEL_FONTSIZE,
        'ytick.labelsize': TICK_LABEL_FONTSIZE,
        'legend.fontsize': LEGEND_FONTSIZE,
        
        'figure.dpi': FIGURE_DPI,
        'axes.labelpad': AXIS_LABEL_PADDING,
        
        'lines.linewidth': LINE_WIDTH,
        'lines.markersize': MARKER_SIZE,
        
        'axes.linewidth': AXIS_LINE_WIDTH,
        'axes.prop_cycle': plt.cycler(color=COLOR_PALETTE),
        'axes.spines.top': True,
        'axes.spines.right': True,
        
        'legend.frameon': True,
        'legend.framealpha': 0.8,
        'legend.edgecolor': 'gray' 
    })
    print("Global Matplotlib styles applied.")
    
# ==============================================================================
#                      CONFIG MANAGEMENT FUNCTIONS
# ==============================================================================

def save_this_config(file_name: str):
    """
    Saves a timestamped copy of the current config.py to the saved_configs directory.
    This is the recommended way to snapshot an experiment's setup.

    Args:
        file_name (str): A descriptive name for the experiment.
        NOTE: It is recommended to include a timestamp in the file name.
    """
    
    os.makedirs(saved_configs_path, exist_ok=True)
    dst_file = os.path.join(saved_configs_path, file_name)
    
    shutil.copy(THIS_FILE_PATH, dst_file)
    
    print(f"Configuration saved as: {file_name} in {saved_configs_path}")     
    
    return dst_file


def load_config_as_object(config_file_path: str):
    """
    Loads a saved Python config file as a module object without overwriting anything.
    This is the safe and correct way to use a saved configuration.

    Args:
        config_file_path (str): The full path to the saved .py config file.

    Returns:
        A module object containing all the variables from the loaded file.
    """
    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"Config file not found: {config_file_path}")

    # Create a unique module name to avoid conflicts
    module_name = f"loaded_config_{os.path.basename(config_file_path).replace('.py', '')}"
    
    # Load the file as a Python module
    spec = importlib.util.spec_from_file_location(module_name, config_file_path)
    loaded_config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded_config)
    
    print(f"Successfully loaded config from: {config_file_path}")
    return loaded_config
    
    
    




