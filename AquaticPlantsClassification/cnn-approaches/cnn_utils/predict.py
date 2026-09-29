# utils/predict.py

import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns

import torch

import a1_cnn_ynlt.config as cfg
import cnn_utils.data as data_utils
import cnn_utils.patching as patching_utils
from cnn_utils.evaluate import EvalClassification

training_data_dir = "/home/takayuki/Desktop/summer2025/plants/data/AquaticPlantLabData/squared" 



class PredictPlants:
    """
    A class for detailed, single-image prediction and analysis, including critical reviews.
    
    This class takes a pre-built, ready-to-use model object and provides methods
    for in-depth analysis beyond simple evaluation metrics.
    """
    def __init__(self,
                 model: torch.nn.Module,
                 model_name: str = "Unnamed Model"):

        self.model = model
        self.model_name = model_name
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval() 

        # --- Configuration from global cfg ---
        self.class_names = cfg.CANONICAL_CLASSNAMES_LIST
        self.class_to_index = cfg.CANONICAL_CLASS_TO_INDEX
        self.invasive_indices = cfg.INVASIVE_INDICES
        self.do_critical_review = cfg.do_critical_review
        self.critical_confusion_level = cfg.critical_confusion_level
        self.PDF_PAGE_SIZE = cfg.PDF_PAGE_SIZE
        
        self.font_sizes = {
            'suptitle': 18,
            'title': 14,
            'body': 11,
            'label': 12,
            'tick': 10,
            'bar_text': 8
        }

    def list_classes(self):
        print("Available classes with indices:")
        max_len = max(len(name) for name in self.class_names) 
        for i, class_name in enumerate(self.class_names):
            print(f"{str(i).ljust(3)}: {class_name.ljust(max_len)}", end=" ")
            if i in self.invasive_indices:
                print("(Invasive Species)")
            else:
                print()
        
    def predict(self, input, true_label=None, display = True, return_fig = False, k = 5, sample_id=None):
        """
        Predict the class of a single input image (PIL/numpy/tensor).
        Args:
            k - for top-k predictions
        Returns a dict containing:
        - predicted label
        - confidence
        - entropy
        - margin
        - top-k predictions
        - matplotlib axes (if display=True)
        - correctness (if true_label is provided)
        """
        input_tensor = data_utils.image_to_tensor(cfg, input, device=self.device)

        self.model.eval()
        with torch.no_grad():
            output = self.model(input_tensor)

        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        predicted_class_idx = torch.argmax(probabilities).item()
        predicted_class_name = self.class_names[predicted_class_idx]
        predicted_confidence = probabilities[predicted_class_idx].item()

        # Confusion metrics
        entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10)).item()
        sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
        margin = sorted_probs[0].item() - sorted_probs[1].item()

        # Top-k predictions
        topk = [(self.class_names[idx], sorted_probs[i].item()) for i, idx in enumerate(sorted_indices[:k])]

        # Evaluate ground truth label (if provided)
        true_label_idx = None
        is_correct = None
        in_topk = None
        
        if true_label is not None:  
            if isinstance(true_label, str):
                true_label_idx = self.class_to_index[true_label] # <-- THE FIX
                in_topk = any(label == true_label for label, _ in topk)
                is_correct = true_label == predicted_class_name
            elif isinstance(true_label, (int, np.integer)):
                label_str = self.class_names[int(true_label)]
                true_label_idx = int(true_label)
                in_topk = any(label == label_str for label, _ in topk)
                is_correct = label_str == predicted_class_name
            elif isinstance(true_label, torch.Tensor):
                true_label_idx = true_label.item()
                in_topk = any(label == self.class_names[true_label_idx] for label, _ in topk)
                is_correct = true_label_idx == predicted_class_idx        
           
            
        # Return dictionary of results
        return_dict = {
            "predicted_class_idx": predicted_class_idx,
            "predicted_class_name": predicted_class_name,
            "confidence": predicted_confidence,
            "entropy": entropy,
            "margin": margin,
            "topk": topk,
            "true_label_idx": true_label_idx,
            "is_correct": is_correct,
            "in_topk": in_topk,
        }
        
        confusion_level = self.how_confused(return_dict)
        return_dict["confusion_level"] = confusion_level
        
        is_critical = self.is_critical(return_dict)
        return_dict["is_critical"] = is_critical
        return_dict["critical_review_info"] = None 
        
                
        if display or return_fig:
            
            # --- Common plotting preparations ---
            img_to_display = data_utils.to_uint8_img(cfg, input_tensor)
            probabilities_np = probabilities.cpu().numpy()
            
            # Define bar colors based on config and logic
            bar_colors = [cfg.PRIMARY_COLOR] * len(self.class_names)
            correct_color = cfg.COLOR_PALETTE[2] # Green
            
            if true_label_idx is not None:
                bar_colors[true_label_idx] = correct_color
                if not is_correct:
                    bar_colors[predicted_class_idx] = cfg.SECONDARY_COLOR
            else:
                bar_colors[predicted_class_idx] = cfg.SECONDARY_COLOR

            # --- Create a single, double-column figure with a precise GridSpec layout ---
            fig = plt.figure(figsize=(cfg.FIG_WIDTH_DOUBLE_COL_INCH, cfg.FIG_HEIGHT_DOUBLE_COL_INCH * 0.5))

            gs = fig.add_gridspec(3, 2, 
                                  width_ratios=[1, 3], 
                                  height_ratios=[2.5, 2.5, 1],
                                  wspace=0.3, hspace=0.1)

            ax_img = fig.add_subplot(gs[0:2, 0])
            ax_text = fig.add_subplot(gs[2, 0])
            ax_hist = fig.add_subplot(gs[0:2, 1])
            
            # --- Panel 1: Image and Title ---
            ax_img.imshow(img_to_display)
            ax_img.axis("off")

            title_parts = [f"Pred: {predicted_class_name}"]
            title_color = cfg.NEUTRAL_COLOR
            if true_label_idx is not None:
                true_label_name = self.class_names[true_label_idx]
                title_parts.insert(0, f"True: {true_label_name}")
                title_color = correct_color if is_correct else cfg.INVASIVE_HIGHLIGHT_COLOR
            
            ax_img.set_title("\n".join(title_parts), color=title_color, size=cfg.TICK_LABEL_FONTSIZE)

            # --- Panel 2: Metrics Text ---
            ax_text.axis("off")
            details_text = (f"Confidence: {predicted_confidence:.2f}\n"
                          f"Margin: {margin:.2f}\n"
                          f"Entropy: {entropy:.2f}\n"
                          f"Confusion Level: {confusion_level}\n"
                          f"Critical?: {'Yes' if is_critical else 'No'}")

            bbox_props = dict(boxstyle='round,pad=0.5',
                              fc=(1, 1, 1, plt.rcParams['legend.framealpha']),
                              ec=plt.rcParams['legend.edgecolor'], 
                              lw=plt.rcParams['axes.linewidth'])

            ax_text.text(0.5, 0.0, details_text, transform=ax_text.transAxes,
                         fontsize=cfg.ANNOTATION_FONTSIZE, va='center', ha='center',
                         bbox=bbox_props)

            # --- Panel 3: Probability Bar Chart ---
            bars = ax_hist.bar(self.class_names, probabilities_np, color=bar_colors)
            ax_hist.set_ylabel("Probability")
            
            ax_hist.set_ylim(0, max(1.0, np.max(probabilities_np)))
            ax_hist.grid(axis='y', linestyle=cfg.GRID_STYLE, alpha=cfg.GRID_ALPHA)
            
            # Use existing config variables for bar labels
            for bar in bars:
                yval = bar.get_height()
                if yval > 0.01:
                    ax_hist.text(bar.get_x() + bar.get_width()/2.0, yval, f" {yval:.2f}",
                                 ha='center', va='bottom',
                                 fontsize=cfg.ANNOTATION_FONTSIZE - 2, # Use a valid config variable
                                 rotation=00) # Reuse existing rotation config

            # Use a hardcoded value for tick padding as no config variable exists
            ax_hist.tick_params(axis='x', 
                                rotation=cfg.DIST_CHART_XTICK_ROTATION,
                                pad=1) # Hardcoded padding for tighter look
                                
            for label in ax_hist.get_xticklabels():
                label.set_ha(cfg.DIST_CHART_XTICK_ALIGNMENT)
                if label.get_text() in cfg.INVASIVE_SPECIES_NAMES:
                    label.set_color(cfg.INVASIVE_HIGHLIGHT_COLOR)
            
            # --- Final Layout and Display ---
            if display:
                plt.show() 
                
            if return_fig:
                return_dict["figure"] = fig
                return_dict["axes"] = {'img': ax_img, 'text': ax_text, 'hist': ax_hist}
            else:
                plt.close(fig)
            
         
        # Display summary table   
        if display:    
            metrics_data = [
                ("Predicted Class", return_dict["predicted_class_name"]),
                ("Confidence", f"{return_dict['confidence']:.4f}"),
                ("Entropy", f"{return_dict['entropy']:.4f}"),
                ("Margin", f"{return_dict['margin']:.4f}"),
                ("Is Correct?", return_dict["is_correct"]),
                ("In Top-k?", return_dict["in_topk"]),
                ("Confusion Level", return_dict["confusion_level"]),
                ("Is Critical?", return_dict["is_critical"])
            ]
            topk_data = return_dict["topk"]
            
            num_rows = max(len(metrics_data), len(topk_data))
            table_data = []
            
            for i in range(num_rows):
                metric_col = metrics_data[i][0] if i < len(metrics_data) else ""
                value_col = metrics_data[i][1] if i < len(metrics_data) else ""
                
                topk_class_col = topk_data[i][0] if i < len(topk_data) else ""
                topk_prob_col = f"{topk_data[i][1]:.4f}" if i < len(topk_data) else ""
                
                table_data.append([metric_col, value_col, topk_class_col, topk_prob_col])
        
            headers = ["Metric", "Value", "Top-k Prediction", "Confidence"]
            print("\nPrediction Summary:")
            print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        
        
        if is_critical and self.do_critical_review:
            
            if display:
                print("\n**CRITICAL PREDICTION**: This prediction requires further review.")
            
            if true_label_idx is None:
                true_label_idx = -1  # If no true label is provided, set to -1
                
            critical_pred = self.critical_review(input_tensor, true_label_idx, 
                                                 return_figure= return_fig, display=display)
            
            if return_fig and "figure" in critical_pred:
                return_dict["critical_review_figure"] = critical_pred["figure"]
            
            final_prediction = critical_pred["final_prediction_class_name"]
            
            # Populate critical review info
            return_dict["critical_review_info"] = {
                "initial_prediction_idx": predicted_class_idx,
                "initial_prediction_name": predicted_class_name,
                "final_prediction_idx": self.class_to_index[final_prediction],
                "final_prediction_name": final_prediction
            }
            
            if display:
                print(f"\nInitial Prediction: {predicted_class_name}")
                print(f"Final Prediction after Critical Review: {final_prediction}")
            
            return_dict["predicted_class_name"] = final_prediction
            return_dict["predicted_class_idx"] = critical_pred["final_prediction_class_index"]
            
            if true_label_idx is not None:
                true_label_name = self.class_names[true_label_idx]
                return_dict["is_correct"] = (final_prediction == true_label_name)
            
            if return_fig and "figure" in critical_pred:
                return_dict["critical_review_figure"] = critical_pred["figure"]
        
        return return_dict
        
    def how_confused(self, prediction_dict):
        """
        Check if the prediction is confused based on the confidence margin and entropy.
        
        A prediction is considered confused if the margin is less than a threshold.
        This threshold can be adjusted based on model performance.
        
        Expected Input:
            prediction_dict: Dictionary returned by the predict method.
        
        Returns:
            Degree of confusion as an integer.
        """
        
        # Decide confusion level
        margin = prediction_dict.get("margin", -1.0)
        entropy = prediction_dict.get("entropy", -1.0)
        confidence = prediction_dict.get("confidence", -1.0)
        
        if -1.0 in [margin, entropy, confidence]:
            raise Warning("Prediction dictionary should contain 'margin', 'entropy', and 'confidence' keys.")
        
        confusion_level = 0
        if margin < cfg.margin_threshold:
            confusion_level += cfg.margin_weight
        if entropy > cfg.entropy_threshold:
            confusion_level += cfg.entropy_weight
        if confidence < cfg.confidence_threshold:
            confusion_level += cfg.confidence_weight
        
        return confusion_level
    
    def is_critical(self, prediction_dict):
        """        
        A prediction is considered critical if the confusion level exceeds a threshold, 
        and the predicted class is not an invasive species.
        This can be useful for identifying predictions that may require further review.
        
        Expected Input:
            prediction_dict: Dictionary returned by the predict method.
        
        Returns:
            True if the prediction is critical, False otherwise.
        """
        
        confusion_level = prediction_dict.get("confusion_level", -1)
        predicted_class_idx = prediction_dict.get("predicted_class_idx", -1)
        
        if -1 in [confusion_level, predicted_class_idx]:
            raise Warning("Prediction dictionary should contain 'confusion_level' and 'predicted_class_idx' keys.")
        
        if confusion_level >= self.critical_confusion_level:
            
            if cfg.ignore_invasive_predictions:
                # only in this case, we may ignore the confusion
                if predicted_class_idx in self.invasive_indices:
                    return False
                
            return True
        
        return False
        
    
    def predict_on_dataloader(self, dataloader, num_preds = None, return_fig = False, display=False, k = 5):
        """
        Runs predictions on a dataloader
        
        Expected Input:
            dataloader: PyTorch DataLoader with images and labels
            num_preds: Number of predictions to run, if None, runs on the entire dataloader
        
        Returns:
            A list of dictionaries with prediction results for each image.
            Each dict contains:
                - predicted_class_idx
                - predicted_class_name
                - confidence
                - entropy
                - margin
                - topk predictions
                - true_label_idx (if available)
                - is_correct (if available)
                - in_topk (if available)
        """
        
        if dataloader.batch_size > 1:
            new_dataloader = torch.utils.data.DataLoader(
                dataloader.dataset,
                batch_size=1,
                shuffle=False
            )
            dataloader = new_dataloader
        
        return_list_of_dicts = []
        
        for image, label in dataloader:
            label_str = self.class_names[label.item()]
            # print(f"Processing image with true label: {label_str}, idx: {label.item()}")
            prediction_info = self.predict(image, true_label=label, return_fig=return_fig, display=display, k = k)
            return_list_of_dicts.append(prediction_info)

            if num_preds is not None and len(return_list_of_dicts) >= num_preds:
                break
                
        return return_list_of_dicts
    
    def critical_review(self, input_tensor, label=None, return_figure=False, display = True):
        """
        Creates a dataloader by patching the input tensor and runs predictions on it.
        This is useful for reviewing critical predictions.
        """
        img = data_utils.to_uint8_img(cfg, input_tensor)
        patch_dataloader = patching_utils.get_dataloader( # Use the new general function
            image=img,
            label=label,
            batch_size=1, 
            shuffle=False
        )
        
        # print("Reviewing critical prediction by patching the image...")
        self.do_critical_review = False  # Disable critical review to avoid recursion
        patch_predictions = self.predict_on_dataloader(patch_dataloader, num_preds=None, return_fig=False, display=False, k=5)
        # print("Review completed. Predictions on patches:")
        
        num_patches = len(patch_predictions)
        
        patch_prediction_scores = {}
        
        for i, pred in enumerate(patch_predictions):
            
            pred_class = pred["predicted_class_name"]
            pred_class_idx = pred["predicted_class_idx"]
            
            max_confusion_level = sum([
                cfg.margin_weight, cfg.entropy_weight, cfg.confidence_weight
            ])
            confusion_level = pred.get("confusion_level", 0)
            
            score = (max_confusion_level - confusion_level) / max_confusion_level
            
            if pred_class not in patch_prediction_scores:
                patch_prediction_scores[pred_class] = 1.0 # Initialize with 1.0 to avoid zero scores
            
            patch_prediction_scores[pred_class] += score
            
        max_score = max(patch_prediction_scores.values())
        max_score_classes = [cls for cls, score in patch_prediction_scores.items() if score == max_score]
        is_tie = list(patch_prediction_scores.values()).count(max_score) > 1
        
        final_prediction = max_score_classes[0]
        
        if is_tie:
            for cls in max_score_classes:
                if cls in cfg.INVASIVE_SPECIES_NAMES:
                    final_prediction = cls
                    break
        
        correct_in_considered_classes = False
        considered_classes = set(patch_prediction_scores.keys())
        true_class = self.class_names[label] if label is not None else None
        
        if true_class is not None and true_class in considered_classes:
            correct_in_considered_classes = True
            
        
        self.do_critical_review = True
        
        return_dict = {
            "final_prediction_class_name": final_prediction,
            "final_prediction_class_index": self.class_to_index[final_prediction],
            "num_patches": num_patches,
            "is_tie": is_tie,
            "correct_in_considered_classes": correct_in_considered_classes,
            "patch_prediction_scores_dict": patch_prediction_scores
            }
        
        if label is not None:
            true_label_name = self.class_names[label]
            return_dict["true_label_class_name"] = true_label_name
            return_dict["true_label_class_index"] = self.class_to_index[true_label_name]
            
            is_correct = final_prediction == true_label_name
            return_dict["is_correct"] = is_correct
          
        if display or return_figure:
            sorted_scores = sorted(patch_prediction_scores.items(), key=lambda item: item[1], reverse=True)
            class_names = [item[0] for item in sorted_scores]
            scores = [item[1] for item in sorted_scores]
            
            true_label_name = self.class_names[label] if label is not None else None
            
            # Define bar colors based on config and logic
            bar_colors = [cfg.PRIMARY_COLOR] * len(class_names)
            correct_color = cfg.COLOR_PALETTE[2] # Green
            
            for i, cls in enumerate(class_names):
                if cls == final_prediction:
                    # If the final prediction is correct, it will be overridden by the next condition
                    bar_colors[i] = cfg.SECONDARY_COLOR # Orange for final prediction
                if true_label_name is not None and cls == true_label_name:
                    bar_colors[i] = correct_color # Green for true label
            
            # --- Figure Generation ---
            # Make figure slightly taller to accommodate rotated labels
            fig, ax = plt.subplots(figsize=(cfg.FIG_WIDTH_SINGLE_COL_INCH, cfg.FIG_HEIGHT_SINGLE_COL_INCH * 1.3))

            # --- KEY CHANGE: Replace tight_layout with subplots_adjust ---
            # This manually sets the margins inside the figure canvas.
            # You may need to tweak these values slightly for a perfect fit.
            fig.subplots_adjust(
                left=0.18,    # Increase left margin for y-axis label
                right=0.98,   # Keep right margin tight
                bottom=0.35,  # Increase bottom margin significantly for rotated labels
                top=0.95      # Keep top margin tight
            )
            # Create the bar plot
            bars = ax.bar(class_names, scores, color=bar_colors)
            
            ax.set_ylabel("Aggregated Score")
            # The x-axis label is often self-evident and can be removed for cleaner manuscript figures
            # ax.set_xlabel("Predicted Class") 
            
            if cfg.SHOW_TITLES_IN_PLOTS:
                title = f"Critical Review: {final_prediction}"
                ax.set_title(title)
            
            # Set y-limit with a bit of padding
            ax.set_ylim(0, max(scores) * 1.15)
            ax.grid(axis='y', linestyle=cfg.GRID_STYLE, alpha=cfg.GRID_ALPHA)

            # Add score text on top of each bar
            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2.0, yval, f"{yval:.2f}",
                        ha='center', va='bottom', fontsize=cfg.ANNOTATION_FONTSIZE)

            # Rotate labels and highlight invasive species
            ax.tick_params(axis='x', rotation=cfg.DIST_CHART_XTICK_ROTATION, labelsize = 10)
            for xtick_label in ax.get_xticklabels():
                xtick_label.set_ha(cfg.DIST_CHART_XTICK_ALIGNMENT)
                if xtick_label.get_text() in cfg.INVASIVE_SPECIES_NAMES:
                    xtick_label.set_color(cfg.INVASIVE_HIGHLIGHT_COLOR)
            
            # Final layout adjustments
            fig.tight_layout(pad=0.5)

            if display:  
                plt.show()
                
            if return_figure:
                return_dict["figure"] = fig
            else:
                plt.close(fig)
            
        if display:            
            # Tabulate the return dict and show
            print("\nCritical Review Summary:")
            summary_table = [
                ("Final Prediction Class", return_dict["final_prediction_class_name"]),
                ("Final Prediction Index", return_dict["final_prediction_class_index"]),
                ("Number of Patches", return_dict["num_patches"]),
                ("Is Tie?", return_dict["is_tie"]),
                ("True Label Class", return_dict.get("true_label_class_name", "N/A")),
                ("True Label Index", return_dict.get("true_label_class_index", "N/A")),
                ("Is Correct?", return_dict.get("is_correct", "N/A")),
            ]
            print(tabulate(summary_table, headers=["Metric", "Value"], tablefmt="grid"))
            
        return return_dict
    
    def evaluate_with_predict_fn(self, dataloader): 
        """
        Initializes an evaluation object using the predict function.
        This is a non-visual method.
        Returns the EvalClassification object.
        """
        evaluator = EvalClassification(
            cfg,
            model=self.model,
            dataloader=dataloader,
        )
        
        # This lambda now explicitly states it does not want figures.
        predict_fn = lambda image, label: self.predict(
            input=image, 
            true_label=label,
            return_fig=False, # Hardcode to False
            display=False, 
            k=5
        )
        evaluator.evaluate(predict_function = predict_fn)
        
        return evaluator
    
                
            
        
        
        

        
        
            
        
            
            
            
            
            
