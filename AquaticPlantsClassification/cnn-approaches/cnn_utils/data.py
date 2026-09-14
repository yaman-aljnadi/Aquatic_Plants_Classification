# cnn_utils/data.py

"""Data loading, processing, and visualization utilities for the Plant Classification project.

The `PlantData` class is for managing data splits and the primary factory functions 
(`get_3_dataloaders`, `get_single_dataloader`) are for creating PyTorch DataLoaders.

Most functions require a configuration object to be passed to them, ensuring that parameters 
like data paths, transformations, and plot styling are handled consistently.

Main Functions:
    get_3_dataloaders:              Creates and returns TRAIN, VALIDATION, and TEST dataloaders.
    get_single_dataloader:          Creates a SINGLE dataloader from a specified directory.
    get_ood_dataloaders:            Creates VALIDATION and TEST dataloaders for out-of-distribution data.

Visualization Functions:
    visualize_dataloader:           Displays a batch of images from any dataloader.
    visualize_augmentations:        Generates a figure showing examples of data augmentations.
    visualize_total_distribution:   Creates a bar chart of the dataset's class distribution.

Helper Functions:
    image_to_tensor:                Converts a PIL image or numpy array into a normalized tensor.
    to_uint8_img:                   Reverses normalization to convert a tensor back to a displayable image.
    get_labels_from_dataloader:     Extracts all labels from a dataloader.
"""

import os
import numpy as np
import torch
from collections import Counter
from tabulate import tabulate
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch
from PIL import Image

from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from sklearn.model_selection import StratifiedKFold, train_test_split

def canonical_class_name(name: str) -> str:
    """Map folder names such as 'Brasenia schreberi (watershield)' to the scientific name."""
    return name.split("(")[0].strip()

def remap_imagefolder_to_canonical(dataset, cfg):
    """Rewrite ImageFolder samples so labels use cfg.CANONICAL_CLASS_TO_INDEX."""
    remapped_samples = []
    skipped = 0
    for path, local_idx in dataset.samples:
        local_name = canonical_class_name(dataset.classes[local_idx])
        if local_name in cfg.CANONICAL_CLASS_TO_INDEX:
            remapped_samples.append((path, cfg.CANONICAL_CLASS_TO_INDEX[local_name]))
        else:
            skipped += 1
            print(f"Warning: Class '{local_name}' not in canonical list. Skipping: {path}")
    if not remapped_samples:
        raise ValueError("No images matched the canonical class list.")
    if skipped:
        print(f"Skipped {skipped} images whose folder names are not canonical classes.")
    dataset.samples = remapped_samples
    dataset.imgs = remapped_samples
    dataset.targets = [s[1] for s in remapped_samples]
    dataset.classes = cfg.CANONICAL_CLASSNAMES_LIST
    dataset.class_to_idx = cfg.CANONICAL_CLASS_TO_INDEX
    return dataset

class PlantData():
    
    def __init__(self, config, data_dir: str = None, batch_size: int = None,
                 train_percent: float = None, val_percent: float = None,
                 test_percent: float = None, random_seed: int = None):

        self.config = config

        #    This makes the class flexible. You can override a config value for a one-off test.
        self.data_dir = data_dir if data_dir is not None else self.config.main_data_dir
        self.batch_size = batch_size if batch_size is not None else self.config.batch_size
        self.train_percent = train_percent if train_percent is not None else self.config.train_percent
        self.val_percent = val_percent if val_percent is not None else self.config.val_percent
        self.test_percent = test_percent if test_percent is not None else self.config.test_percent
        self.random_seed = random_seed if random_seed is not None else self.config.random_seed

        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        self.dataset_init = remap_imagefolder_to_canonical(
            datasets.ImageFolder(self.data_dir), self.config
        )
        self.all_indices = list(range(len(self.dataset_init)))
        self.all_targets = self.dataset_init.targets
        
    
    def calc_split_sizes(self):
        total_size = len(self.dataset_init)
        train_size = int(total_size * self.train_percent)
        test_size = int(total_size * self.test_percent)
        val_size = total_size - train_size - test_size
        
        print(f"Total dataset size \t: {total_size}")
        print(f"Calculated train size \t: {train_size}")
        print(f"Calculated val size \t: {val_size}")
        print(f"Calculated test size \t: {test_size}\n")
        
    
    def get_dataloaders(self):
        
        if self.test_percent == 1.0:
            print("Using the entire dataset as the test set. No train/val split.")
            
            # 1. Let ImageFolder discover files and classes in the test directory.
            #    This dataset might have the wrong class indices.
            raw_dataset = remap_imagefolder_to_canonical(
                datasets.ImageFolder(self.data_dir, transform=self.config.basic_tf),
                self.config,
            )
            print(f"Found {len(raw_dataset.classes)} canonical classes.")
            
            # Now we can correctly calculate the size
            self.calc_split_sizes()
            
            self.test_dataset = raw_dataset
            test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
            return None, None, test_dataloader
        
        train_val_indices, test_indices, train_val_targets, test_targets = train_test_split(
            self.all_indices, self.all_targets,
            test_size=self.test_percent,
            stratify=self.all_targets,
            random_state=self.random_seed
        )
        
        relative_val_percent = self.val_percent / (self.train_percent + self.val_percent)
        
        train_indices, val_indices, _, _ = train_test_split(
            train_val_indices,
            train_val_targets,
            test_size=relative_val_percent,
            stratify=train_val_targets,
            random_state=self.random_seed
        )
        
        dataset_train_augmented = remap_imagefolder_to_canonical(
            datasets.ImageFolder(self.data_dir, transform=self.config.aug_tf), self.config
        )
        dataset_val_basic = remap_imagefolder_to_canonical(
            datasets.ImageFolder(self.data_dir, transform=self.config.basic_tf), self.config
        )
        dataset_test_basic = remap_imagefolder_to_canonical(
            datasets.ImageFolder(self.data_dir, transform=self.config.basic_tf), self.config
        )

        self.train_dataset = Subset(dataset_train_augmented, train_indices)
        self.val_dataset = Subset(dataset_val_basic, val_indices)
        self.test_dataset = Subset(dataset_test_basic, test_indices)

        train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
        
        return train_dataloader, val_dataloader, test_dataloader
    
    def get_split_info(self, classwise: bool = True):
        
        if self.test_percent == 1.0:
            print("Using the entire dataset as the test set. No train/val split.")
            self.calc_split_sizes()
            return
        
        if not (hasattr(self, 'train_dataset') and hasattr(self, 'val_dataset') and hasattr(self, 'test_dataset')):
            print("Can't get split info. DataLoaders are not initialized. Call get_dataloaders() first.")
            return
        
        def count_classes(subset):
            labels = [subset.dataset.targets[i] for i in subset.indices]
            label_counts = Counter(labels)
            return label_counts

        train_class_counts = count_classes(self.train_dataset)
        val_class_counts = count_classes(self.val_dataset)
        test_class_counts = count_classes(self.test_dataset)

        if classwise:
            class_names = self.dataset_init.classes
            headers = ["Class Name", "Train Count", "Validation Count", "Test Count"]
            table_data = []

            for i, class_name_str in enumerate(class_names):
                train_count = train_class_counts.get(i, 0)
                val_count = val_class_counts.get(i, 0)
                test_count = test_class_counts.get(i, 0)
                table_data.append([class_name_str, train_count, val_count, test_count])

            print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        print(f" Total dataset size \t: {len(self.dataset_init)}")
        print(f" Train dataset size \t: {len(self.train_dataset)}")
        print(f" Val dataset size \t: {len(self.val_dataset)}")
        print(f" Test dataset size \t: {len(self.test_dataset)}\n")
    
    def unnormalize_image(self, tensor_image):
        """Reverses the normalization for visualization."""
        if tensor_image.ndim == 4: 
            tensor_image = tensor_image[0]
        
        mean = torch.tensor(self.config.normalization_parameters["mean"], dtype=tensor_image.dtype, device=tensor_image.device).view(3, 1, 1)
        std = torch.tensor(self.config.normalization_parameters["std"], dtype=tensor_image.dtype, device=tensor_image.device).view(3, 1, 1)
        
        img = tensor_image.clone()
        for t, m, s in zip(img, mean, std):
            t.mul_(s).add_(m)
        img = torch.clamp(img, 0, 1)
        return img

    def visualize_dataloader(self, dataloader, num_images=4):
        """
        Visualize images from the dataloader, fetching multiple batches if necessary.
        """
        
        current_batch_size = dataloader.batch_size
        collected_images = []
        collected_labels = []
        dataloader_iter = iter(dataloader) 

        while len(collected_images) < num_images:
            try:
                batch_images, batch_labels = next(dataloader_iter)
                for i in range(batch_images.shape[0]):
                    if len(collected_images) < num_images:
                        collected_images.append(batch_images[i])
                        collected_labels.append(batch_labels[i])
                    else:
                        break 
            except StopIteration:
                print("Dataloader exhausted before collecting the desired number of images.")
                break 

        if not collected_images:
            print("No images were collected to display.")
            return

        actual_num_to_display = len(collected_images)

        fig_width = 5 * actual_num_to_display
        fig_height = 5
        if actual_num_to_display == 1:
            fig_width = 5

        fig, axes = plt.subplots(1, actual_num_to_display, figsize=(fig_width, fig_height), squeeze=False)

        for i in range(actual_num_to_display):
            image = self.unnormalize_image(collected_images[i]).permute(1, 2, 0).cpu().numpy()
            ax = axes[0, i]
            ax.imshow(image)
            ax.set_title(self.config.CANONICAL_INDEX_TO_CLASS[collected_labels[i].item()])
            ax.axis('off')

        plt.tight_layout()
        plt.show()
   

def get_single_dataloader(cfg, data_dir = None):
        
    train_percent = 0.0
    val_percent = 0.0
    test_percent = 1.0

    batch_size = 1

    test_data_obj = PlantData(cfg, data_dir, batch_size, train_percent, val_percent, test_percent)

    _, _, test_dataloader = test_data_obj.get_dataloaders()
    
    print(f"Test DataLoader created with {len(test_dataloader.dataset)} images from {data_dir}.")
    return test_dataloader
   
def get_3_dataloaders(cfg, data_dir = None, batch_size = None,
                      train_percent = None, val_percent = None,
                      test_percent = None, random_seed = None):
    """
    Get train, validation, and test dataloaders using a config object.
    """
    plant_data_obj = PlantData(cfg, data_dir, batch_size,
                               train_percent, val_percent, test_percent,
                               random_seed)

    train_dataloader, val_dataloader, test_dataloader = plant_data_obj.get_dataloaders()
    plant_data_obj.get_split_info(classwise=False)
    return train_dataloader, val_dataloader, test_dataloader

def get_ood_dataloaders(cfg,
                ood_data_dir = None,
                val_percent = 0.5,
                test_percent = 0.5,
                batch_size = 1,
                random_seed = None,
                ):
    
    if ood_data_dir is None:
        ood_data_dir = cfg.hand_test_data_dir  
    if random_seed is None:
        random_seed = cfg.random_seed
        
    raw_dataset = remap_imagefolder_to_canonical(
        datasets.ImageFolder(ood_data_dir), cfg
    )
    print(f"Found {len(raw_dataset.classes)} canonical classes in OOD data.")

    remapped_samples = raw_dataset.samples
    
    all_indices = list(range(len(remapped_samples)))
    all_targets = [s[1] for s in remapped_samples]

    try:
        val_ood_indices, test_ood_indices, _, _ = train_test_split(
            all_indices,
            all_targets,
            test_size=test_percent,
            stratify=all_targets,
            random_state=random_seed
        )
    except ValueError:
        print("OOD stratify failed (a class has too few images). Using an unstratified split.")
        val_ood_indices, test_ood_indices, _, _ = train_test_split(
            all_indices,
            all_targets,
            test_size=test_percent,
            random_state=random_seed
        )
    
    base_ood_dataset = datasets.ImageFolder(ood_data_dir, transform=cfg.basic_tf)
    base_ood_dataset.samples = remapped_samples
    base_ood_dataset.imgs = remapped_samples  
    base_ood_dataset.targets = all_targets
    base_ood_dataset.classes = cfg.CANONICAL_CLASSNAMES_LIST
    base_ood_dataset.class_to_idx = cfg.CANONICAL_CLASS_TO_INDEX

    val_ood_dataset = Subset(base_ood_dataset, val_ood_indices)
    test_ood_dataset = Subset(base_ood_dataset, test_ood_indices)
    
    print(f"\nOOD split complete:")
    print(f"  OOD Validation set size: {len(val_ood_dataset)}")
    print(f"  OOD Test set size    : {len(test_ood_dataset)}\n")

    # 5. Create the DataLoaders
    val_ood_dataloader = DataLoader(val_ood_dataset, batch_size=batch_size, shuffle=False)
    test_ood_dataloader = DataLoader(test_ood_dataset, batch_size=batch_size, shuffle=False)
    
    return val_ood_dataloader, test_ood_dataloader

def image_to_tensor(cfg, image, device=None):
    """
    Convert an image to a tensor suitable for model input.
    Supports PIL Images or numpy arrays.
    Expected Input:
        PIL Image, numpy array (HWC), or torch Tensor (N=1, C, H, W) or Tensor (C, H, W).
    Output:
        Torch tensor of shape (N=1, C, 224, 224) ready for model prediction.
    """
    if isinstance(image, Image.Image):
        if image.mode != "RGB":
            image = image.convert("RGB")
        # Pass PIL Image directly to transform pipeline
        tensor = cfg.basic_tf(image)
    
    elif isinstance(image, np.ndarray):
        if image.ndim == 2: # Grayscale 
            image = np.stack([image]*3, axis=-1)
            
        # Convert numpy HWC to PIL image for basic_tf (which expects PIL or tensor)
        pil_image = Image.fromarray(image)
        tensor = cfg.basic_tf(pil_image)
    
    elif isinstance(image, torch.Tensor):
        if device is None:
            device = image.device
            
        tensor = image
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
            
        tensor = tensor.to(device)  
    
    else:
        raise ValueError("Input image must be a PIL Image, numpy array, or torch Tensor.")
    
    # Ensure batch dimension
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
        
    if tensor.ndim != 4:
        raise ValueError("Input tensor must have 4 dimensions (N, C, H, W).")
    elif tensor.shape[0] > 1:
        raise ValueError("We expect a batch size of 1 in the prediction phase.")

    if tensor.shape[1] != 3 or tensor.shape[2:] != (224, 224):
        raise ValueError("Input tensor must have shape (N, 3, 224, 224) where N is the batch size.")
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'        

    return tensor.to(device)

def to_uint8_img(cfg, tensor_image):
    """
    Reverses the normalization for visualization.
    
    Expected Input: 
        Tensor of shape (N=1, C, H, W)
    Output:
        Numpy array for visualization of shape (H, W, C) in [0, 255], unit 8 format.
    """
        
    mean = torch.tensor(cfg.normalization_parameters["mean"], dtype=tensor_image.dtype, device=tensor_image.device).view(3, 1, 1)
    std = torch.tensor(cfg.normalization_parameters["std"], dtype=tensor_image.dtype, device=tensor_image.device).view(3, 1, 1)
    
    img = tensor_image.clone()
    
    img = img * std + mean  
    img = torch.clamp(img, 0, 1) # Ensure values are [0, 1]
    img = img.squeeze(0)  # Remove batch dimension
    img = img.permute(1, 2, 0).cpu().numpy() # Convert to HWC format
    
    img = (img * 255).astype(np.uint8)  # Convert to [0, 255] range and uint8 format
    
    return img

def visualize_dataloader(cfg, dataloader, class_names=None, unnormalize_fn = to_uint8_img, num_images=4):
    """
    Visualizes images from a dataloader. This function is general and can work
    with any dataloader that yields (images, labels) tuples.

    Args:
        dataloader (DataLoader): The dataloader to visualize.
        class_names (list): A list of strings for class labels.
        unnormalize_fn (function): A function that takes a normalized tensor image
                                   and returns a displayable image (e.g., a numpy array).
        num_images (int): The maximum number of images to display.
    """
    collected_images = []
    collected_labels = []
    dataloader_iter = iter(dataloader)

    while len(collected_images) < num_images:
        try:
            batch_images, batch_labels = next(dataloader_iter)
            for i in range(len(batch_images)):
                if len(collected_images) < num_images:
                    collected_images.append(batch_images[i])
                    collected_labels.append(batch_labels[i])
                else:
                    break
        except StopIteration:
            print(f"Dataloader exhausted at {len(collected_images)} images collected.")
            break

    if not collected_images:
        print("No images were collected to display.")
        return

    actual_num_to_display = len(collected_images)
    cols = min(actual_num_to_display, 4) # Max 4 columns
    rows = (actual_num_to_display + cols - 1) // cols # Calculate required rows
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), squeeze=False)
    axes = axes.flatten() 

    for i in range(actual_num_to_display):
        display_image = unnormalize_fn(cfg, collected_images[i])
        label_index = collected_labels[i].item()

        ax = axes[i]
        ax.imshow(display_image)
        ax.set_title(class_names[label_index] if class_names else f"Image {i+1}\nClass {label_index}", fontsize = 35)
        ax.axis('off')

    for j in range(actual_num_to_display, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

def visualize_augmentations(cfg, dataloader, num_examples=9, cols=5, save_filename: str = None, display=True):
    """
    Generates a figure showcasing the effect of various data augmentations in a wide grid.

    This function picks a single image and applies a selection of augmentations,
    displaying the results in a grid with a user-specified number of columns.
    The figure is styled according to the global config.py for manuscript-readiness.

    Args:
        dataloader (DataLoader): Dataloader to source an original image from.
        num_examples (int): The number of augmented examples to show.
        cols (int): The number of columns in the output figure grid.
        save_filename (str): Filename to save the figure (e.g., 'fig_augmentations.pdf').
        display (bool): Whether to show the plot interactively.
    """
    # --- 1. Select Representative Augmentations and an Original Image ---
    augmentations_to_show = {
        "Horizontal Flip": transforms.v2.RandomHorizontalFlip(p=1.0),
        "Vertical Flip": transforms.v2.RandomVerticalFlip(p=1.0),
        "Rotation": transforms.v2.RandomRotation(degrees=(30, 60)),
        "Perspective": transforms.v2.RandomPerspective(distortion_scale=0.4, p=1.0),
        "Color Jitter": transforms.v2.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
        "Resized Crop": transforms.v2.RandomResizedCrop(size=(224, 224), scale=(0.2, 0.4)),
        "Close-up Crop": transforms.v2.RandomResizedCrop(size=(224, 224), scale=(0.08, 0.1)), # <-- NEWLY ADDED LINE
        "Trivial Augment": transforms.v2.TrivialAugmentWide(),
        "Random Erasing": transforms.v2.Compose([
            transforms.v2.ToTensor(),
            transforms.v2.RandomErasing(p=1.0, scale=(0.05, 0.15)),
            transforms.ToPILImage()
        ])
    }
    
    if num_examples > len(augmentations_to_show):
        print(f"Warning: Requested {num_examples} examples, but only {len(augmentations_to_show)} are defined. Showing all.")
        num_examples = len(augmentations_to_show)

    try:
        if isinstance(dataloader.dataset, Subset):
            original_dataset = dataloader.dataset.dataset
            sample_path, _ = original_dataset.samples[dataloader.dataset.indices[0]]
        else:
            original_dataset = dataloader.dataset
            sample_path, _ = original_dataset.samples[0]
            
        original_image = Image.open(sample_path).convert("RGB")
        original_image = transforms.Resize((224, 224))(original_image)
    except (AttributeError, IndexError) as e:
        print(f"Error: Could not retrieve a raw image from the dataloader. {e}")
        return

    # --- 2. Create and Populate the Figure ---
    total_panels = num_examples + 1
    rows = (total_panels + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cfg.FIG_WIDTH_DOUBLE_COL_INCH * 1.2, rows * 2.0))
    axes = axes.flatten()

    # Panel 0: The Original Image
    axes[0].imshow(original_image)
    axes[0].set_title("Original", fontsize=cfg.TITLE_FONTSIZE)
    axes[0].axis('off')

    # Subsequent Panels: Augmentations
    selected_aug_keys = list(augmentations_to_show.keys())[:num_examples]

    for i, key in enumerate(selected_aug_keys):
        ax = axes[i + 1]
        transform = augmentations_to_show[key]
        augmented_image = transform(original_image.copy())
        
        ax.imshow(augmented_image)
        ax.set_title(key, fontsize=cfg.TITLE_FONTSIZE)
        ax.axis('off')

    for j in range(total_panels, len(axes)):
        axes[j].axis('off')

    fig.tight_layout(pad=0.5)

    # --- 3. Save and/or Display the Figure ---
    if save_filename:
        full_path = os.path.join(cfg.FIGURE_SAVE_DIR, save_filename)
        try:
            fig.savefig(full_path, dpi=cfg.FIGURE_DPI, bbox_inches='tight')
            print(f"Augmentation figure saved to: {full_path}")
        except Exception as e:
            print(f"Error saving figure: {e}")
    
    if display:
        plt.show()

    plt.close(fig)

def get_labels_from_dataloader(dataloader):
    """Helper function to extract all labels from a dataloader."""
    if not hasattr(dataloader, 'dataset'):
        raise ValueError("Dataloader does not have a 'dataset' attribute.")
    
    dataset = dataloader.dataset
    
    if isinstance(dataset, Subset):
        # Efficiently get labels from a Subset
        original_targets = np.array(dataset.dataset.targets)
        return original_targets[dataset.indices].tolist()
    
    elif hasattr(dataset, 'targets'):
        # Get labels from a full dataset like ImageFolder
        return dataset.targets
        
    else:
        # Fallback for custom datasets: iterate through the dataloader
        print("Warning: Dataset has no 'targets' or 'indices' attribute. Iterating to get labels (this may be slow).")
        all_labels = []
        for _, labels in dataloader:
            all_labels.extend(labels.cpu().numpy())
        return all_labels

class HandlerSplitColorPatch(HandlerPatch):
    def __init__(self, color1, color2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.color1 = color1
        self.color2 = color2

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        # Create the first half of the patch (non-invasive color)
        patch1 = mpatches.Rectangle([xdescent, ydescent], width / 2, height,
                                    facecolor=self.color1, edgecolor='none')
        # Create the second half of the patch (invasive color)
        patch2 = mpatches.Rectangle([xdescent + width / 2, ydescent], width / 2, height,
                                    facecolor=self.color2, edgecolor='none')
        return [patch1, patch2]
    
def visualize_total_distribution(cfg, dataloader, class_names, save_filename: str = None, display = True):
    """
    Generates a simple, clean bar chart of the total class distribution for a dataset.

    This function creates a publication-ready figure, sorted by sample count, with all
    styling (fonts, colors, titles, etc.) controlled by the central config.py file.
    It's designed to be called from a master script like figure_generation.ipynb.

    Args:
        dataloader (DataLoader): The DataLoader for the dataset to visualize.
        class_names (list): A list of strings for the class names.
        save_filename (str): The filename to save the figure (e.g., 'fig1_total_dist.pdf').
                             The figure is always saved and the plot is closed to manage memory.
    """
    # --- 1. Data Preparation ---
    labels = get_labels_from_dataloader(dataloader)
    counts = Counter(labels)
    distribution_data = [(class_names[i], counts.get(i, 0)) for i in range(len(class_names))]
    sorted_distribution = sorted(distribution_data, key=lambda item: item[1], reverse=True)
    
    sorted_class_names = [item[0] for item in sorted_distribution]
    sorted_counts = [item[1] for item in sorted_distribution]
    
    invasive_species = getattr(cfg, 'INVASIVE_SPECIES_NAMES', [])

    # --- 2. Plotting Setup  ---
    fig, ax = plt.subplots(figsize=cfg.DIST_CHART_FIGSIZE)
    colors = [cfg.DIST_INVASIVE_COLOR if name in invasive_species else cfg.DIST_NON_INVASIVE_COLOR for name in sorted_class_names]

    # --- 3. Create and Style the Plot  ---
    bars = ax.bar(sorted_class_names, sorted_counts, color=colors)
    
    # Use config for label padding
    vertical_offset = max(sorted_counts) * cfg.BAR_LABEL_PADDING_FACTOR if sorted_counts else 0
    
    for bar in bars:
        yval = bar.get_height()
        if yval > 0:
            ax.text(bar.get_x() + bar.get_width() / 2.0, yval + vertical_offset, int(yval),
                    ha='center', va='bottom', size=cfg.ANNOTATION_FONTSIZE)

    ax.set_ylabel("Number of Images", labelpad=cfg.DIST_CHART_YLABEL_PADDING)
    
    # Use config for tick label styling
    plt.setp(ax.get_xticklabels(), rotation=cfg.DIST_CHART_XTICK_ROTATION, ha=cfg.DIST_CHART_XTICK_ALIGNMENT)
    
    if sorted_counts:
        # Use config for y-axis top margin
        ax.set_ylim(0, max(sorted_counts) + 5)

    # Use config for font weight and color
    for label in ax.get_xticklabels():
        if label.get_text() in invasive_species:
            label.set_color(cfg.INVASIVE_HIGHLIGHT_COLOR)
        else:
            label.set_color(cfg.NEUTRAL_COLOR)

    if cfg.SHOW_TITLES_IN_PLOTS:
        ax.set_title("Total Image Distribution")

    # Use config for legend labels and location
    invasive_patch = mpatches.Patch(color=cfg.DIST_INVASIVE_COLOR, label=cfg.LEGEND_LABEL_INVASIVE)
    non_invasive_patch = mpatches.Patch(color=cfg.DIST_NON_INVASIVE_COLOR, label=cfg.LEGEND_LABEL_NON_INVASIVE)
    ax.legend(handles=[invasive_patch, non_invasive_patch], loc=cfg.DIST_CHART_LEGEND_LOCATION)

    # Use config for grid style
    ax.grid(axis='y', linestyle=cfg.GRID_STYLE, alpha=cfg.GRID_ALPHA)
    fig.tight_layout()

    # --- 4. Save and Close  ---
    if save_filename:
        full_path = os.path.join(cfg.FIGURE_SAVE_DIR, save_filename)
        
        try:
            # NOTE: The config file already creates FIGURE_SAVE_DIR, so no need for os.makedirs
            fig.savefig(full_path, dpi=cfg.FIGURE_DPI, bbox_inches='tight')
            print(f"Figure successfully saved to: {full_path}")
        except Exception as e:
            print(f"Error saving figure to {full_path}: {e}")
    else:
        print("Warning: No save_filename provided. Figure not saved.")

    if display:
        plt.show()
    
    plt.close(fig)
    
    

def get_repeated_split_seeds(n_repeats=5, base_seed=8):
    """Seeds for repeated 60/20/20 splits. Prefer this over 5-fold: several species have only 2 images."""
    return [base_seed + i for i in range(n_repeats)]


def make_batch_size_one(dataloader):
    """
    Converts the given DataLoader to a batch size of 1.
    
    Args:
        dataloader (DataLoader): The original DataLoader.
        
    Returns:
        DataLoader: A new DataLoader with batch size set to 1.
    """
    return DataLoader(dataloader.dataset, batch_size=1, shuffle=False, num_workers=dataloader.num_workers)
   
        
        
        