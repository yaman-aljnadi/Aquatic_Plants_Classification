# cnn_utils/evaluate.py

import os
import numpy as np
from tqdm import tqdm
from tabulate import tabulate
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns

import timm
import torch
from sklearn.metrics import confusion_matrix, accuracy_score

class EvalClassification:
    """
    This class takes a configuration object, a pre-built model, and a dataloader
    to compute and visualize a comprehensive set of performance metrics. It relies
    on the provided config object for canonical data (like class names) and for
    styling parameters in generated plots.
    
    """
    def __init__(self, cfg,
                 model: torch.nn.Module,
                 dataloader: torch.utils.data.DataLoader):

        self.cfg = cfg
        self.model = model
        self.dataloader = dataloader
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        # --- Standard setup ---
        if self.dataloader.batch_size != 1:
            print("Warning: Forcing dataloader batch size to 1 for evaluation.")
            self.dataloader = torch.utils.data.DataLoader(
                self.dataloader.dataset, batch_size=1, shuffle=False
            )
        
        self.class_names = self.cfg.CANONICAL_CLASSNAMES_LIST
        self.invasive_indices = self.cfg.INVASIVE_INDICES
        
        self.all_predictions = []
        self.all_true_labels = []

    def change_dataloader(self, new_dataloader):
        self.dataloader = new_dataloader
        self.all_predictions = []
        self.all_true_labels = []
        
        print("Data loader updated. Please run evaluate() again to compute new predictions and true labels.")
    
    
    def evaluate(self, predict_function=None):
        """
        Evaluate the model on the provided dataloader.
        If a custom predict function is provided, use it to get predictions.
        The predict function should take a batch of images and return predictions.
            - Note that we constrain the batch size to be 1 for the current evaluation and predict functions.
            - The predict function should return a prediction dictionary with 'predicted_class_idx' as the key for the predicted class index.
        If no predict function is provided, use the model's forward method to get predictions.
        """
        self.all_predictions = []
        self.all_true_labels = []
        
        self.model.eval()  
        with torch.no_grad():
            for _, (images, labels) in tqdm(enumerate(self.dataloader), total=len(self.dataloader), desc="  -- Evaluating"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                label_item = labels.item() 
                
                if predict_function is None:
                    outputs = self.model(images)    
                    _, predictions = torch.max(outputs, 1)
                    
                    self.all_predictions.extend(predictions.cpu().numpy())
                    self.all_true_labels.extend(labels.cpu().numpy())
                    
                else:
                    pred_dict = predict_function(images, label_item)
                    predictions = pred_dict['predicted_class_idx']
                    
                    self.all_predictions.append(predictions)
                    self.all_true_labels.append(label_item)

        return self.all_true_labels, self.all_predictions
    
    def check_eval(self):
        if not hasattr(self, 'all_true_labels') or not hasattr(self, 'all_predictions'):
            print("Please run evaluate() first to compute predictions and true labels.")
            return False
        
        if len(self.all_true_labels) == 0 or len(self.all_predictions) == 0:
            print("No predictions or true labels found. Please check your evaluation data.")
            return False
        
        return True
    
    def get_accuracy(self, verbose=True):
        if not self.check_eval():
            return None   
        self.accuracy = accuracy_score(self.all_true_labels, self.all_predictions)
           
        if verbose:  
            print(f"Accuracy: {self.accuracy:.4f}")
        
        return self.accuracy
    
    def get_classwise_confusion_matrix(self, return_fig=False, display=True, figsize = None):
        if not self.check_eval():
            return None   
        
        num_all_classes = len(self.class_names)
        self.confusion_matrix = confusion_matrix(
            self.all_true_labels, 
            self.all_predictions, 
            labels=list(range(num_all_classes))
        )

        fig = None
         
        if display or return_fig:
            
            default_figsize = (self.cfg.FIG_WIDTH_DOUBLE_COL_INCH, self.cfg.FIG_WIDTH_DOUBLE_COL_INCH * 0.8)
            final_figsize = figsize or default_figsize
            fig, ax = plt.subplots(figsize=final_figsize)

            sns.heatmap(self.confusion_matrix,
                        annot=True,
                        fmt='d',
                        cmap=self.cfg.CMAP_BLUES,
                        xticklabels=self.class_names,
                        yticklabels=self.class_names,
                        ax=ax,
                        # For a dense matrix, the standard annotation size is appropriate.
                        annot_kws={'size': self.cfg.ANNOTATION_FONTSIZE})

            # --- Cleaned up Tick Label Styling ---
            # rcParams handles the font size; we only set color and weight.
            for ticklabel in ax.get_xticklabels():
                if ticklabel.get_text() in self.cfg.INVASIVE_SPECIES_NAMES:
                    ticklabel.set_color(self.cfg.INVASIVE_HIGHLIGHT_COLOR)
                else:
                    ticklabel.set_color(self.cfg.NEUTRAL_COLOR)

            for ticklabel in ax.get_yticklabels():
                if ticklabel.get_text() in self.cfg.INVASIVE_SPECIES_NAMES:
                    ticklabel.set_color(self.cfg.INVASIVE_HIGHLIGHT_COLOR)
                else:
                    ticklabel.set_color(self.cfg.NEUTRAL_COLOR)

            plt.xlabel("Predicted Labels")
            plt.ylabel("True Labels")

            if self.cfg.SHOW_TITLES_IN_PLOTS:
                plt.title("Classwise Confusion Matrix")

            fig.tight_layout()

        if display:
            plt.show()

        if return_fig:
            return self.confusion_matrix, fig

        if not display and not return_fig and fig is not None:
            plt.close(fig)

        return self.confusion_matrix
    
    def get_classwise_metrics(self, display=True):
        if not self.check_eval():
            return None   

        if not hasattr(self, 'confusion_matrix'):
            num_classes = len(self.class_names)
            self.confusion_matrix = confusion_matrix(
                self.all_true_labels, 
                self.all_predictions, 
                labels=list(range(num_classes)) 
            )
        
        self.classwise_data = []        
        self.classwise_data_headers = ["Class", "Support", "TP", "FN", "FP", 
                                       "Precision", "Recall (TPR)", "FNR", "F1-Score", "Specificity (TNR)", "FPR"]
        classwise_precisions = []
        classwise_recalls = []
        classwise_fnrs = []
        classwise_f1s = []
        classwise_specificities = []
        classwise_fprs = []
        classwise_supports = []
        
        for i, class_name in enumerate(self.class_names):
            TP = self.confusion_matrix[i, i]
            FP = self.confusion_matrix[:, i].sum() - TP
            FN = self.confusion_matrix[i, :].sum() - TP
            TN = self.confusion_matrix.sum() - (TP + FP + FN)
            support = TP + FN

            precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
            recall_tpr = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            fnr = FN / (FN + TP) if (FN + TP) > 0 else 0.0
            f1 = 2 * (precision * recall_tpr) / (precision + recall_tpr) if (precision + recall_tpr) > 0 else 0.0
            specificity_tnr = TN / (TN + FP) if (TN + FP) > 0 else 0.0
            fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0
            
            self.classwise_data.append([
                class_name, 
                support, 
                TP, 
                FN, 
                FP, 
                precision, 
                recall_tpr, 
                fnr,
                f1, 
                specificity_tnr, 
                fpr
            ])
            
            classwise_precisions.append(precision)
            classwise_recalls.append(recall_tpr)
            classwise_fnrs.append(fnr)
            classwise_f1s.append(f1)
            classwise_specificities.append(specificity_tnr)
            classwise_fprs.append(fpr)
            classwise_supports.append(support)
        
        macro_precision = np.mean(classwise_precisions)
        macro_recall = np.mean(classwise_recalls)
        macro_fnr = np.mean(classwise_fnrs)
        macro_f1 = np.mean(classwise_f1s)
        macro_specificity = np.mean(classwise_specificities) 
        macro_fpr = np.mean(classwise_fprs) 

        
        if sum(classwise_supports) > 0:
            weighted_precision = np.average(classwise_precisions, weights=classwise_supports)
            weighted_recall = np.average(classwise_recalls, weights=classwise_supports)
            weighted_fnr = np.average(classwise_fnrs, weights=classwise_supports)
            weighted_f1 = np.average(classwise_f1s, weights=classwise_supports)
            weighted_specificity = np.average(classwise_specificities, weights=classwise_supports) 
            weighted_fpr = np.average(classwise_fprs, weights=classwise_supports) 

        summary_headers = ["Metric", "Macro", "Weighted"]
        summary_data = [
            ["Precision", macro_precision, weighted_precision],
            ["Recall (TPR)", macro_recall, weighted_recall],
            ["FNR", macro_fnr, weighted_fnr],
            ["F1-Score", macro_f1, weighted_f1],
            ["Specificity (TNR)", macro_specificity, weighted_specificity],
            ["FPR", macro_fpr, weighted_fpr],
            ["Accuracy",   self.get_accuracy(verbose=False), self.get_accuracy(verbose=False)]
        ]
        
        
        if display:
            print("\nPer-Class Metrics:")
            print(tabulate(self.classwise_data, headers=self.classwise_data_headers, tablefmt="grid"))
        
            print("\nSummary Metrics:")
            print(tabulate(summary_data, headers=summary_headers, tablefmt="grid"))
            
        self.classwise_metrics = {
            "classwise_data": self.classwise_data,
            "classwise_headers": self.classwise_data_headers,
            "summary_data": summary_data,
            "summary_headers": summary_headers
        }
        
        self.classwise_summary = {}
        for metric in summary_data:
            self.classwise_summary[metric[0]] = {
                "Macro": metric[1],
                "Weighted": metric[2]
            }
        
        self.classwise_metrics['classwise_summary'] = self.classwise_summary
        
        return self.classwise_metrics
    

    def get_binary_confusion_matrix(self, return_fig=False, display=True, figsize = None):
        if not self.check_eval():
            return None   
        
        binary_true_labels = [1 if label in self.invasive_indices else 0 for label in self.all_true_labels]
        binary_predictions = [1 if pred in self.invasive_indices else 0 for pred in self.all_predictions]
        self.binary_confusion_matrix = confusion_matrix(binary_true_labels, binary_predictions)
        
        fig = None 
        if display or return_fig: 
            
            default_figsize = (self.cfg.FIG_WIDTH_SINGLE_COL_INCH, self.cfg.FIG_WIDTH_SINGLE_COL_INCH)
            final_figsize = figsize or default_figsize
            fig, ax = plt.subplots(figsize=final_figsize)
            
            sns.heatmap(self.binary_confusion_matrix,
                    annot=True,
                    fmt='d',
                    cmap=self.cfg.CMAP_BLUES,
                    xticklabels=['Non-Invasive', 'Invasive'],
                    yticklabels=['Non-Invasive', 'Invasive'],
                    ax=ax,
                    cbar=False,
                    annot_kws={'size': self.cfg.ANNOTATION_FONTSIZE_LARGE})
            
            plt.xlabel("Predicted Labels")
            plt.ylabel("True Labels")

            
            if self.cfg.SHOW_TITLES_IN_PLOTS:
                plt.title("Binary Confusion Matrix")

            fig.tight_layout()

        if display:
            plt.show()

        if return_fig:
            return self.binary_confusion_matrix, fig

        if not display and not return_fig and fig is not None:
            plt.close(fig)

        return self.binary_confusion_matrix
    
    def get_binary_metrics(self, display=True):
        
        if not hasattr(self, 'binary_confusion_matrix'):
            self.get_binary_confusion_matrix(display=False)
        
        cm_binary = self.binary_confusion_matrix
        
        # Positive = Invasive, Negative = Non-Invasive

        TN = cm_binary[0, 0] # True Non-Invasive predicted as Non-Invasive
        FP = cm_binary[0, 1] # True Non-Invasive predicted as Invasive
        FN = cm_binary[1, 0] # True Invasive predicted as Non-Invasive
        TP = cm_binary[1, 1] # True Invasive predicted as Invasive

        fpr = FP / (FP + TN) # False Positive Rate - proportion of Non-Invasive species incorrectly classified as Invasive
        fnr = FN / (TP + FN) # False Negative Rate - proportion of Invasive species incorrectly classified as Non-Invasive {this is important!}
        tnr = TN / (TN + FP) # True Negative Rate (Specificity) - proportion of Non-Invasive species correctly classified
        tpr = TP / (TP + FN) # True Positive Rate (Recall or Sensitivity) - proportion of Invasive species correctly classified

        ppv = TP / (TP + FP) # Positive Predictive Value (Precision) - proportion of predicted Invasive species that are actually Invasive
        npv = TN / (TN + FN) # Negative Predictive Value - proportion of Non-Invasive predictions that are actually Non-Invasive

        f1 = 2 * (ppv * tpr) / (ppv + tpr) # F1-Score (Harmonic mean of Precision and Recall) - how well the model balances precision and recall 
        overall_accuracy = (TN + TP) / cm_binary.sum() # Overall correct classification rate

        metrics_data = [
            ["True Positives (TP)", TP],
            ["True Negatives (TN)", TN],
            ["False Positives (FP)", FP],
            ["False Negatives (FN)", FN],
            ["False Positive Rate (FPR)", f"{fpr:.4f}"],
            ["False Negative Rate (FNR)", f"{fnr:.4f}"],
            ["(TNR or Specificity)", f"{tnr:.4f}"],
            ["(TPR or Sensitivity or Recall)", f"{tpr:.4f}"],
            ["Negative Predictive Value (NPV)", f"{npv:.4f}"],
            ["Positive \" (PPV or Precision)", f"{ppv:.4f}"],
            ["F1 Score", f"{f1:.4f}"],
            ["Overall Accuracy", f"{overall_accuracy:.4f}"]
        ]

        headers = ["Confusion Matrix", "Values", "Rates", "Metrics", "Predictive Values & Overall", "Metrics"]
        metrics_col_wise = []
        num_metrics = len(metrics_data)
        rows_per_col = (num_metrics + 2) // 3 
        for r in range(rows_per_col):
            new_row = []
            for c in range(3): 
                metric_index = r + c * rows_per_col
                if metric_index < num_metrics:
                    new_row.extend(metrics_data[metric_index])
                else:
                    new_row.extend(["", ""]) 
            metrics_col_wise.append(new_row)

        if display:
            print("\nBinary Classification Metrics:")
            print("Positive = Invasive & Negative = Non-Invasive")
            print(tabulate(metrics_col_wise, headers=headers, tablefmt="grid"))
        
        metrics_dict = {
            "TP": TP,
            "TN": TN,
            "FP": FP,
            "FN": FN,
            "FPR": fpr,
            "FNR": fnr,
            "TNR": tnr,
            "TPR": tpr,
            "PPV": ppv,
            "NPV": npv,
            "F1": f1,
            "Accuracy": overall_accuracy
        }
            
        return metrics_dict
    
    def print_definitions(self):
        print("\nDefinitions of Metrics:")
        definitions = {
            "TP": "True Positives - Invasive species correctly classified as Invasive",
            "TN": "True Negatives - Non-Invasive species correctly classified as Non-Invasive",
            "FP": "False Positives - Non-Invasive species incorrectly classified as Invasive",
            "FN": "False Negatives - Invasive species incorrectly classified as Non-Invasive",
            "FPR": "False Positive Rate - Proportion of Non-Invasive species incorrectly classified as Invasive",
            "FNR": "False Negative Rate - Proportion of Invasive species incorrectly classified as Non-Invasive {IMPORTANT!}",
            "TNR": "True Negative Rate (Specificity) - Proportion of Non-Invasive species correctly classified",
            "TPR": "True Positive Rate (Recall or Sensitivity) - Proportion of Invasive species correctly classified",
            "PPV": "Positive Predictive Value (Precision) - Proportion of predicted Invasive species that are actually Invasive",
            "NPV": "Negative Predictive Value - Proportion of Non-Invasive predictions that are actually Non-Invasive",
            "F1 Score": "Harmonic mean of Precision and Recall, balancing both metrics",
            "Overall Accuracy": "Overall correct classification rate"
        }
        max_key_len = max(len(key) for key in definitions.keys())
        for key, value in definitions.items():
            key.ljust(max_key_len)
            print(f"{key}:  {value}")
    
    def evaluate_accuracy(self, dataloader=None):
        """
        Evaluate the model's accuracy on the provided dataloader.
        If no dataloader is provided, use the current dataloader.
        """
        if dataloader is not None:
            self.change_dataloader(dataloader)
        
        self.evaluate()
        return self.get_accuracy()
    
    def evaluate_binary_metrics(self, dataloader=None):
        """
        Evaluate binary classification metrics (Invasive vs Non-Invasive).
        If no dataloader is provided, use the current dataloader.
        """
        if dataloader is not None:
            self.change_dataloader(dataloader)
        
        self.evaluate()
        return self.get_binary_metrics()
    
    def evaluate_classwise_metrics(self, dataloader=None, predict_function=None):
        """
        Evaluate classwise metrics for all classes.
        If no dataloader is provided, use the current dataloader.
        """
        if dataloader is not None:
            self.change_dataloader(dataloader)
        
        self.evaluate(predict_function=predict_function)
        return self.get_classwise_metrics()
    
    def visualize_classwise_metrics(self, sort_by='name', return_fig=False, display=True, figsize = None):
        """
        Creates a diverging bar chart to visualize per-class accuracy vs. FNR.

        This visualization is useful for quickly identifying classes that are
        difficult for the model to correctly classify (high FNR) and those
        it performs well on (high Accuracy/Recall).

        - 'Accuracy' is represented by Recall (TPR), as it's the standard
          per-class accuracy metric (TP / (TP + FN)).
        - FNR (False Negative Rate) is shown on the negative side to represent error.
        - Invasive species are highlighted in red for easy identification.

        Args:
            sort_by (str): How to sort the classes on the y-axis.
                           Options: 'name' (alphabetical), 'accuracy' (descending),
                           'fnr' (descending).
            return_fig (bool): If True, returns the matplotlib figure object.
            display (bool): If True, shows the plot.
        """
        if not self.check_eval():
            return None
        
        if not hasattr(self, 'classwise_data'):
            self.get_classwise_metrics(display=False)

        # Extract data needed for plotting # Column indices: 0=Class, 6=Recall, 7=FNR
        plot_data = [(row[0], row[6], row[7]) for row in self.classwise_data]

        if sort_by == 'accuracy':
            plot_data.sort(key=lambda x: x[1], reverse=True)
        elif sort_by == 'fnr':
            plot_data.sort(key=lambda x: x[2], reverse=True)

        class_names = [item[0] for item in plot_data]
        recalls = np.array([item[1] for item in plot_data])
        fnrs = np.array([item[2] for item in plot_data])

        num_classes = len(class_names)
        dynamic_height = (self.cfg.BAR_CHART_BASE_HEIGHT_INCH + 
                          (num_classes * self.cfg.BAR_CHART_HEIGHT_PER_ITEM_INCH))
        
        default_figsize = (self.cfg.FIG_WIDTH_DOUBLE_COL_INCH, dynamic_height)
        final_figsize = figsize or default_figsize
        fig, ax = plt.subplots(figsize=final_figsize)

        # --- Plotting and Styling ---
        y_pos = np.arange(num_classes)
        
        # REVISED: Using PRIMARY/SECONDARY colors and the new BAR_CHART_THICKNESS.
        ax.barh(y_pos, recalls, align='center', height=self.cfg.BAR_CHART_THICKNESS,
                color=self.cfg.PRIMARY_COLOR, label='Accuracy (Recall)')
        ax.barh(y_pos, -fnrs, align='center', height=self.cfg.BAR_CHART_THICKNESS,
                color=self.cfg.SECONDARY_COLOR, label='FNR (Miss Rate)')

        for i, (recall, fnr) in enumerate(zip(recalls, fnrs)):
            # Annotate the 'recall' bar
            ax.annotate(f'{recall:.2f}', 
                        xy=(recall, i), 
                        xytext=(3, 0), 
                        textcoords='offset points',
                        size=self.cfg.ANNOTATION_FONTSIZE,
                        va='center', 
                        ha='left')

            # Annotate the 'fnr' bar
            ax.annotate(f'{fnr:.2f}', 
                        xy=(-fnr, i), 
                        xytext=(-3, 0), 
                        textcoords='offset points',
                        size=self.cfg.ANNOTATION_FONTSIZE,
                        va='center', 
                        ha='right')
            
        # REFACTOR: Using config variables for font weights.
        ax.set_yticks(y_pos)
        ax.set_yticklabels(class_names)
        for ticklabel in ax.get_yticklabels():
            if ticklabel.get_text() in self.cfg.INVASIVE_SPECIES_NAMES:
                ticklabel.set_color(self.cfg.INVASIVE_HIGHLIGHT_COLOR)
            else:
                ticklabel.set_color(self.cfg.NEUTRAL_COLOR)
        
        ax.invert_yaxis()
        ax.axvline(0, color=self.cfg.AXIS_GUIDELINE_COLOR, linewidth=self.cfg.AXIS_GUIDELINE_WIDTH)
        ax.set_xlabel("Metric Value")

        if self.cfg.SHOW_TITLES_IN_PLOTS:
            ax.set_title("Class-wise Performance: Accuracy (Recall) vs. FNR")

        ax.legend(loc='upper left')
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{abs(x):.1f}'))
        
        xlim_val = 1.0 + self.cfg.BAR_CHART_X_AXIS_PADDING
        ax.set_xlim([-xlim_val, xlim_val])
        
        fig.tight_layout()

        if display:
            plt.show()
        if return_fig:
            return fig
        if not display and not return_fig:
            plt.close(fig)
            