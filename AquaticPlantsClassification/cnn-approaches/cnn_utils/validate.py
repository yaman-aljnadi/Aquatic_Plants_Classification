# utils/validate.py

import os
from datetime import datetime
from tqdm import tqdm
from tabulate import tabulate
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import torch

# --- Project-specific imports ---
import a1_cnn_ynlt.config as cfg
from cnn_utils.evaluate import EvalClassification
from cnn_utils.predict import PredictPlants 

class ReportGenerator:
    """
    Generates a multi-page PDF report for a SINGLE model, including
    baseline vs. critical review analysis.
    """
    def __init__(self, predictor: PredictPlants, datasets_to_test: dict, experiment_name: str):
        self.predictor = predictor
        self.datasets_to_test = datasets_to_test
        self.experiment_name = experiment_name
        
        self.experiment_name = cfg.experiment_name
        report_artifact_name = f"{self.experiment_name}_{self.predictor.model_name}"
        
        self.pdf_path = os.path.join(cfg.EXPERIMENTS_DIR, f"{report_artifact_name}.pdf")
        self.saved_config_path = cfg.save_this_config(f"{report_artifact_name}_config.py")
        
        self.results = {}
        self.evaluators = {}
        

    def _run_single_test(self, predictor: PredictPlants, dataloader: torch.utils.data.DataLoader):
        """
        Runs the Baseline vs. Critical Review test and now also aggregates
        detailed statistics about the critical review outcomes.
        """
        # --- Part 1: Run Baseline Evaluation ---
        print(f"\n  - Running baseline evaluation for {predictor.model_name}...")
        baseline_evaluator = EvalClassification(cfg, model=predictor.model, dataloader=dataloader)
        baseline_evaluator.evaluate()

        # --- Part 2: Run Critical Review Evaluation & Collect Stats ---
        print(f"\n  - Running critical review evaluation for {predictor.model_name}...")
        
        # Initialize counters for our new metrics
        critical_review_stats = {
            'total_reviews': 0,
            'correct_to_correct': 0,
            'correct_to_wrong': 0,
            'wrong_to_correct': 0,
            'wrong_to_wrong': 0
        }
        
        # We need to loop manually to inspect the detailed output of `predict`
        critical_preds, critical_true_labels = [], []
        predictor.model.eval()
        with torch.no_grad():
            for image, label in tqdm(dataloader, desc="  -> Critical Review"):
                true_idx = label.item()
                
                # Use the enhanced predict function
                pred_dict = predictor.predict(image, true_label=true_idx, display=False, return_fig=False)
                
                # Aggregate final predictions for overall accuracy calculation
                critical_preds.append(pred_dict['predicted_class_idx'])
                critical_true_labels.append(true_idx)

                if pred_dict.get("is_critical") and pred_dict.get("critical_review_info"):
                    info = pred_dict["critical_review_info"]
                    critical_review_stats['total_reviews'] += 1
                    
                    initial_correct = (info['initial_prediction_idx'] == true_idx)
                    final_correct = (info['final_prediction_idx'] == true_idx)
                    
                    if initial_correct and final_correct:
                        critical_review_stats['correct_to_correct'] += 1
                    elif initial_correct and not final_correct:
                        critical_review_stats['correct_to_wrong'] += 1
                    elif not initial_correct and final_correct:
                        critical_review_stats['wrong_to_correct'] += 1
                    elif not initial_correct and not final_correct:
                        critical_review_stats['wrong_to_wrong'] += 1

        critical_reviews_successful = critical_review_stats["correct_to_correct"] + critical_review_stats["wrong_to_correct"]
        critical_review_stats['success_rate'] = (critical_reviews_successful / critical_review_stats['total_reviews']) if critical_review_stats['total_reviews'] > 0 else 0
        critical_review_stats['effectiveness'] = (critical_review_stats["wrong_to_correct"] / critical_review_stats['total_reviews']) if critical_review_stats['total_reviews'] > 0 else 0
        
        # Create a new EvalClassification object just for the final metrics
        # This is cleaner than using the internal lists of the baseline evaluator
        critical_evaluator = EvalClassification(cfg, model=predictor.model, dataloader=dataloader)
        critical_evaluator.all_predictions = critical_preds
        critical_evaluator.all_true_labels = critical_true_labels

        # --- Part 3: Compile all results ---
        results = {
            'baseline_accuracy': baseline_evaluator.get_accuracy(verbose=False),
            'critical_accuracy': critical_evaluator.get_accuracy(verbose=False),
            'baseline_fnr': baseline_evaluator.get_binary_metrics(display=False).get('FNR'),
            'critical_fnr': critical_evaluator.get_binary_metrics(display=False).get('FNR'),
            'critical_review_stats': critical_review_stats 
        }
        
        return results, baseline_evaluator, critical_evaluator

    def run_analysis(self):
        """Runs the analysis for the single model across all datasets."""
        print(f"--- Starting Analysis for Model: {self.predictor.model_name} ---")
        for data_name, dataloader in self.datasets_to_test.items():
            print(f"\n  - On dataset: {data_name}")
            results, base_eval, crit_eval = self._run_single_test(self.predictor, dataloader)
            self.results[data_name] = results
            self.evaluators[data_name] = {'baseline': base_eval, 'critical': crit_eval}
        print("\n--- Analysis Complete ---")

    def generate_pdf_report(self, dataloader_for_per_sample: torch.utils.data.DataLoader = None):
        """Generates the complete PDF report."""
        if not self.results:
            self.run_analysis()
            
        print(f"\nGenerating PDF report at: {self.pdf_path}")
        with PdfPages(self.pdf_path) as pdf:
            self._add_title_page(pdf)
            self._add_model_config_page(pdf) # New method for model config
            self._add_experiment_config_page(pdf)
            self._add_summary_page(pdf)
            
            for data_name, eval_pair in self.evaluators.items():
                self._add_detailed_metrics_pages(pdf, data_name, eval_pair)

            if dataloader_for_per_sample:
                print("Generating per-sample analysis pages...")
                self._add_per_sample_pages(pdf, dataloader_for_per_sample)

        print("--- PDF Report Generation Complete ---")
        
    # --- Page Generation Methods ---
    
    def _add_title_page(self, pdf):
        title_text = (f"Analysis Report for: {self.predictor.model_name}\n\n"
                      f"Experiment: {self.experiment_name}\n"
                      f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        pdf.savefig(self._create_text_page(title_text, "Model Performance Report"))
        plt.close()

    def _add_model_config_page(self, pdf):
        """Adds a page with the specific model's configuration."""
        model = self.predictor.model
        config_data = list(model.config.items())
        
        report_str = "Model Architecture Configuration\n" + "="*50 + "\n\n"
        report_str += f"Model Class: {model.__class__.__name__}\n\n"
        report_str += tabulate(config_data, headers=["Parameter", "Value"], tablefmt="grid")
        
        pdf.savefig(self._create_text_page(report_str, "Model Configuration"))
        plt.close()

    def _add_experiment_config_page(self, pdf):
        import inspect
        report_str = "Key Configuration Parameters:\n" + "=" * 50 + "\n\n"
        
        config_vars = {name: value for name, value in inspect.getmembers(cfg) if not name.startswith('__') and not inspect.ismodule(value)}
        groups = cfg.report_config_groups
        for group_name, var_list in groups.items():
            report_str += f"--- {group_name} ---\n"
            for var_name in var_list:
                if var_name in config_vars:
                    value = config_vars[var_name]
                    value_str = f"[{value[0]}, ..., {value[-1]}] (Total: {len(value)})" if isinstance(value, list) and len(value) > 5 else str(value)
                    report_str += f"{var_name:<30}: {value_str}\n"
            report_str += "\n"
        
        
        report_str += "--- Report Configuration ---\n"
        report_str += f"{'Configuration Snapshot':<30}: {os.path.basename(self.saved_config_path)}\n\n"
        
        pdf.savefig(self._create_text_page(report_str, "Experiment Configuration")); plt.close()

    def _add_summary_page(self, pdf):
        """Adds the summary page with baseline vs critical review stats."""
        # This is your old _add_ab_summary_page, simplified for one model
        report_str = f"Performance Summary: Baseline vs. Critical Review\n"
        report_str += "=" * 60 + "\n\n"
            
        for data_name, results in self.results.items():
            report_str += f"--- Analysis on: {data_name} ---\n"
            
            # --- Main Performance Table ---
            b_acc = f"{results.get('baseline_accuracy', 0):.4f}"
            c_acc = f"{results.get('critical_accuracy', 0):.4f}"
            b_fnr = f"{results.get('baseline_fnr', 0) * 100:.2f}%"
            c_fnr = f"{results.get('critical_fnr', 0) * 100:.2f}%"
            
            table = [
                ["Metric", "Baseline", "With Critical Review"],
                ["Overall Accuracy", b_acc, c_acc],
                ["Invasive FNR*", b_fnr, c_fnr]
            ]
            report_str += tabulate(table, headers="firstrow", tablefmt="grid")
            report_str += "\n*FNR (False Negative Rate): % of actual Invasive plants predicted as Non-Invasive. Lower is better.\n"
            
            # Did cr reduce FNR?
            if results.get('baseline_fnr', 0) > results.get('critical_fnr', 0):
                report_str += f"Critical Review reduced  binary FNR for Invasive plants by {results.get('baseline_fnr', 0) - results.get('critical_fnr', 0):.4f}.\n\n"
            else:
                report_str += "Critical Review did not reduce  binary FNR for Invasive plants.\n\n"
            
            # --- Critical Review Stats Table ---
            stats = results.get('critical_review_stats', {})
            if stats.get('total_reviews', 0) > 0:
                stats_table = [
                    ["Metric", "Count"],
                    ["Total Critical Reviews", stats.get('total_reviews', 0)],
                    ["Correct -> Correct", stats.get('correct_to_correct', 0)],
                    ["Correct -> Wrong (Harmful)", stats.get('correct_to_wrong', 0)],
                    ["Wrong -> Correct (Helpful)", stats.get('wrong_to_correct', 0)],
                    ["Wrong -> Wrong", stats.get('wrong_to_wrong', 0)],
                    ["Success Rate", f"{stats.get('success_rate', 0) * 100:.2f}%"],
                    ["Effectiveness", f"{stats.get('effectiveness', 0) * 100:.2f}%"]
                ]
                report_str += "Critical Review Breakdown:\n"
                report_str += tabulate(stats_table, headers="firstrow", tablefmt="grid")
                report_str += "\n"
            
            report_str += "-"*60 + "\n\n" 
        
        pdf.savefig(self._create_text_page(report_str, "Performance Summary"))
        plt.close()

    def _add_detailed_metrics_pages(self, pdf, data_name, eval_pair):
        """
        Adds detailed metric pages for a given model and dataset, including
        confusion matrices and formatted metric tables.
        """
        title = f"Detailed Metrics: on {data_name}"
        pdf.savefig(self._create_text_page(f"Detailed breakdown for:\n\nDataset: {data_name}", title))
        plt.close()
        
        for eval_type, evaluator in eval_pair.items():
            # --- Page 1: Class-wise Confusion Matrix & Metrics Table ---
            
            # Get the figure and the data for the metrics table
            _, cm_fig = evaluator.get_classwise_confusion_matrix(display=False, return_fig=True)
            classwise_results = evaluator.get_classwise_metrics(display=False)
            
            # Format the tables into a string
            summary_table_str = tabulate(classwise_results['summary_data'], headers=classwise_results['summary_headers'], tablefmt="grid")
            classwise_table_str = tabulate(classwise_results['classwise_data'], headers=classwise_results['classwise_headers'], tablefmt="grid")
            
            # Create a text page for the metrics
            metrics_text = f"Overall Summary Metrics:\n{summary_table_str}\n\nPer-Class Metrics:\n{classwise_table_str}"
            page_title = f"{eval_type.title()} - Class-wise Metrics"
            
            # Add the pages to the PDF
            cm_fig.suptitle(page_title) # Add a title to the figure
            pdf.savefig(cm_fig)
            plt.close(cm_fig)
            pdf.savefig(self._create_text_page(metrics_text, page_title))
            plt.close()

            # --- Page 2: Binary Confusion Matrix & Metrics Table ---
            
            # Get the figure and the data for the metrics table
            _, bin_cm_fig = evaluator.get_binary_confusion_matrix(display=False, return_fig=True)
            binary_metrics_dict = evaluator.get_binary_metrics(display=False)
            
            # Format the dictionary into a two-column table string
            binary_table_data = list(binary_metrics_dict.items())
            binary_table_str = tabulate(binary_table_data, headers=["Metric", "Value"], tablefmt="grid")
            
            # Create a text page for the metrics
            binary_metrics_text = f"Binary Classification Metrics (Invasive vs. Non-Invasive):\n\n{binary_table_str}"
            page_title = f"{eval_type.title()} - Binary Metrics"
            
            # Add the pages to the PDF
            bin_cm_fig.suptitle(page_title) # Add a title to the figure
            pdf.savefig(bin_cm_fig)
            plt.close(bin_cm_fig)
            pdf.savefig(self._create_text_page(binary_metrics_text, page_title))
            plt.close()

    def _add_per_sample_pages(self, pdf, dataloader):
        """Adds a page for each sample, now including the tabulated details."""
        title = f"Per-Sample Analysis: {self.predictor.model_name}"
        pdf.savefig(self._create_text_page(f"Detailed analysis for individual samples.", title))
        plt.close()
        
        for i, (image, label) in tqdm(enumerate(dataloader), desc="Generating sample pages"):
            pred_info = self.predictor.predict(input=image, true_label=label, return_fig=True, display=False, k=5, sample_id=i + 1)
            
            if "figure" in pred_info:
                pdf.savefig(pred_info["figure"])
                plt.close(pred_info["figure"])

            details_text = self._format_prediction_details(pred_info, i + 1)
            pdf.savefig(self._create_text_page(details_text, f"Details for Sample #{i+1}"))
            plt.close()

            # Add critical review figure if it exists
            if pred_info.get("is_critical") and "critical_review_figure" in pred_info:
                pdf.savefig(pred_info["critical_review_figure"])
                plt.close(pred_info["critical_review_figure"])
                
    def _format_prediction_details(self, pred_info, sample_id):
        """Formats the prediction dictionary into a string with two tables."""
        text = f"Prediction Summary for Sample #{sample_id}\n" + "="*50 + "\n\n"
        
        metrics_data = [
            ("Predicted Class", pred_info["predicted_class_name"]),
            ("True Class", pred_info.get("true_label_name", "N/A")),
            ("Is Correct?", pred_info["is_correct"]),
            ("Confidence", f"{pred_info['confidence']:.4f}"),
            ("Is Critical?", pred_info["is_critical"]),
            ("Confusion Level", pred_info["confusion_level"]),
        ]
        text += "Overall Metrics:\n"
        text += tabulate(metrics_data, headers=["Metric", "Value"], tablefmt="grid")
        text += "\n\nTop-5 Predictions:\n"
        
        topk_data = [(name, f"{prob:.4f}") for name, prob in pred_info["topk"]]
        text += tabulate(topk_data, headers=["Class", "Probability"], tablefmt="grid")
        
        return text
    
    @staticmethod
    def _create_text_page(text_content: str, title: str):
        fig = plt.figure(figsize=cfg.PDF_PAGE_SIZE)
        fig.clf()
        fig.suptitle(title, fontsize=16)
        fig.text(0.05, 0.9, text_content, transform=fig.transFigure, size=10, ha="left", va="top", fontfamily="monospace")
        return fig


# class ReportGeneratorAB:
#     """
#     Generates a multi-page PDF report comparing two models (A and B)
#     across multiple datasets. This class orchestrates the evaluation and formatting.
#     """
#     def __init__(self, predictor_a: PredictPlants, predictor_b: PredictPlants, datasets_to_test: dict, experiment_name: str):
#         """
#         Initializes the ReportGenerator.

#         Args:
#             predictor_a (PredictPlants): An initialized PredictPlants instance for Model A.
#             predictor_b (PredictPlants): An initialized PredictPlants instance for Model B.
#             datasets_to_test (dict): A dictionary like {'Dataset Name': dataloader}.
#             experiment_name (str): The name for the experiment, used in the output filename.
#         """
#         self.predictor_a = predictor_a
#         self.predictor_b = predictor_b
#         self.datasets_to_test = datasets_to_test
#         self.experiment_name = experiment_name
        
#         timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
#         report_artifact_name = f"report_{self.experiment_name}_{timestamp}"
        
#         os.makedirs(cfg.EXPERIMENTS_DIR, exist_ok=True)
#         self.pdf_path = os.path.join(cfg.EXPERIMENTS_DIR, f"{report_artifact_name}.pdf")
#         self.saved_config_path = cfg.save_this_config(f"{report_artifact_name}_config.py")
        
        
#         self.comparison_results = {}
#         self.evaluators = {}

#     def _run_single_test(self, predictor: PredictPlants, dataloader: torch.utils.data.DataLoader):
#         """
#         Runs the Baseline vs. Critical Review test and now also aggregates
#         detailed statistics about the critical review outcomes.
#         """
#         # --- Part 1: Run Baseline Evaluation ---
#         print(f"\n  - Running baseline evaluation for {predictor.model_name}...")
#         baseline_evaluator = EvalClassification(model=predictor.model, dataloader=dataloader)
#         baseline_evaluator.evaluate()

#         # --- Part 2: Run Critical Review Evaluation & Collect Stats ---
#         print(f"\n  - Running critical review evaluation for {predictor.model_name}...")
        
#         # Initialize counters for our new metrics
#         critical_review_stats = {
#             'total_reviews': 0,
#             'correct_to_correct': 0,
#             'correct_to_wrong': 0,
#             'wrong_to_correct': 0,
#             'wrong_to_wrong': 0
#         }
        
#         # We need to loop manually to inspect the detailed output of `predict`
#         critical_preds, critical_true_labels = [], []
#         predictor.model.eval()
#         with torch.no_grad():
#             for image, label in tqdm(dataloader, desc="  -> Critical Review"):
#                 true_idx = label.item()
                
#                 # Use the enhanced predict function
#                 pred_dict = predictor.predict(image, true_label=true_idx, display=False, return_fig=False)
                
#                 # Aggregate final predictions for overall accuracy calculation
#                 critical_preds.append(pred_dict['predicted_class_idx'])
#                 critical_true_labels.append(true_idx)

#                 if pred_dict.get("is_critical") and pred_dict.get("critical_review_info"):
#                     info = pred_dict["critical_review_info"]
#                     critical_review_stats['total_reviews'] += 1
                    
#                     initial_correct = (info['initial_prediction_idx'] == true_idx)
#                     final_correct = (info['final_prediction_idx'] == true_idx)
                    
#                     if initial_correct and final_correct:
#                         critical_review_stats['correct_to_correct'] += 1
#                     elif initial_correct and not final_correct:
#                         critical_review_stats['correct_to_wrong'] += 1
#                     elif not initial_correct and final_correct:
#                         critical_review_stats['wrong_to_correct'] += 1
#                     elif not initial_correct and not final_correct:
#                         critical_review_stats['wrong_to_wrong'] += 1

#         critical_reviews_successful = critical_review_stats["correct_to_correct"] + critical_review_stats["wrong_to_correct"]
#         critical_review_stats['success_rate'] = (critical_reviews_successful / critical_review_stats['total_reviews']) if critical_review_stats['total_reviews'] > 0 else 0
#         critical_review_stats['effectiveness'] = (critical_review_stats["wrong_to_correct"] / critical_review_stats['total_reviews']) if critical_review_stats['total_reviews'] > 0 else 0
        
#         # Create a new EvalClassification object just for the final metrics
#         # This is cleaner than using the internal lists of the baseline evaluator
#         critical_evaluator = EvalClassification(model=predictor.model, dataloader=dataloader)
#         critical_evaluator.all_predictions = critical_preds
#         critical_evaluator.all_true_labels = critical_true_labels

#         # --- Part 3: Compile all results ---
#         results = {
#             'baseline_accuracy': baseline_evaluator.get_accuracy(verbose=False),
#             'critical_accuracy': critical_evaluator.get_accuracy(verbose=False),
#             'baseline_fnr': baseline_evaluator.get_binary_metrics(display=False).get('FNR'),
#             'critical_fnr': critical_evaluator.get_binary_metrics(display=False).get('FNR'),
#             'critical_review_stats': critical_review_stats 
#         }
        
#         return results, baseline_evaluator, critical_evaluator

#     def run_ab_comparison(self):
#         """
#         Orchestrates the A/B testing across all models and datasets.
#         """
#         print("Starting A/B comparison for all models and datasets...")
        
#         models_to_test = {
#             self.predictor_a.model_name: self.predictor_a,
#             self.predictor_b.model_name: self.predictor_b
#         }

#         for model_name, predictor in models_to_test.items():
#             print(f"\n==========================================================")
#             print(f"RUNNING ANALYSIS FOR: {model_name}")
#             print(f"==========================================================")
            
#             self.comparison_results[model_name] = {}
#             self.evaluators[model_name] = {}

#             for data_name, dataloader in self.datasets_to_test.items():
#                 print(f"  - On dataset: {data_name}")
#                 results, baseline_eval, critical_eval = self._run_single_test(predictor, dataloader)
#                 self.comparison_results[model_name][data_name] = results
#                 self.evaluators[model_name][data_name] = {'baseline': baseline_eval, 'critical': critical_eval}
                
#         print("\n...A/B comparison complete.")

#     def generate_pdf_report(self, dataloader_for_per_sample: torch.utils.data.DataLoader = None):
#         """
#         Generates the complete PDF report after running the comparison.

#         Args:
#             dataloader_for_per_sample (torch.utils.data.DataLoader, optional): 
#                 If provided, detailed per-sample analysis pages will be generated
#                 for Model A using this specific dataloader.
#         """
#         if not self.comparison_results:
#             self.run_ab_comparison()
            
#         print(f"\nGenerating PDF report at: {self.pdf_path}")

#         with PdfPages(self.pdf_path) as pdf:
#             # Add pages sequentially
#             self._add_title_page(pdf)
#             self._add_config_page(pdf)
#             self._add_ab_summary_page(pdf)
            
#             # Add detailed pages for each model and dataset
#             for model_name, data_evals in self.evaluators.items():
#                 for data_name, eval_pair in data_evals.items():
#                     self._add_detailed_metrics_pages(pdf, model_name, data_name, eval_pair)

#             # Optional: Add per-sample analysis pages for Model A on the provided dataloader.
#             if dataloader_for_per_sample:
#                 print(f"Generating per-sample analysis for Model A...")
#                 self._add_per_sample_pages(pdf, self.predictor_a, dataloader_for_per_sample)

#         print("PDF report generation complete.")
        
#     # --- Page Generation Methods ---
    
#     def _add_title_page(self, pdf):
#         title_text = f"Aquatic Plant Classification Report\n\nExperiment: {self.experiment_name}\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
#         pdf.savefig(self._create_text_page(title_text, "Analysis Report")); plt.close()

#     def _add_config_page(self, pdf):
#         import inspect
#         report_str = "Key Configuration Parameters:\n" + "=" * 50 + "\n\n"
        
#         config_vars = {name: value for name, value in inspect.getmembers(cfg) if not name.startswith('__') and not inspect.ismodule(value)}
#         groups = cfg.report_config_groups
#         for group_name, var_list in groups.items():
#             report_str += f"--- {group_name} ---\n"
#             for var_name in var_list:
#                 if var_name in config_vars:
#                     value = config_vars[var_name]
#                     value_str = f"[{value[0]}, ..., {value[-1]}] (Total: {len(value)})" if isinstance(value, list) and len(value) > 5 else str(value)
#                     report_str += f"{var_name:<30}: {value_str}\n"
#             report_str += "\n"
        
        
#         report_str += "--- Report Configuration ---\n"
#         report_str += f"{'Configuration Snapshot':<30}: {os.path.basename(self.saved_config_path)}\n\n"
        
#         pdf.savefig(self._create_text_page(report_str, "Experiment Configuration")); plt.close()

#     def _add_ab_summary_page(self, pdf):
#         """
#         Adds a separate summary page for each model to the PDF report.
#         """
#         print("  - Adding A/B summary pages to PDF...")
        
#         # Loop through each model's results
#         for model_desc, model_results in self.comparison_results.items():
            
#             # Start a new string for this model's page content
#             report_str = f"A/B Test Summary for: {model_desc}\n"
#             report_str += "=" * (len(report_str) + 5) + "\n\n"
            
#             # Loop through each dataset this model was tested on
#             for data_name, results in model_results.items():
#                 report_str += f"--- Analysis on: {data_name} ---\n"
                
#                 # --- Main Performance Table ---
#                 b_acc = f"{results.get('baseline_accuracy', 0):.4f}"
#                 c_acc = f"{results.get('critical_accuracy', 0):.4f}"
#                 b_fnr = f"{results.get('baseline_fnr', 0) * 100:.2f}%"
#                 c_fnr = f"{results.get('critical_fnr', 0) * 100:.2f}%"
                
#                 table = [
#                     ["Metric", "Baseline", "With Critical Review"],
#                     ["Overall Accuracy", b_acc, c_acc],
#                     ["Invasive FNR*", b_fnr, c_fnr]
#                 ]
#                 report_str += tabulate(table, headers="firstrow", tablefmt="grid")
#                 report_str += "\n*FNR (False Negative Rate): % of actual Invasive plants predicted as Non-Invasive. Lower is better.\n"
                
#                 # Did cr reduce FNR?
#                 if results.get('baseline_fnr', 0) > results.get('critical_fnr', 0):
#                     report_str += f"Critical Review reduced  binary FNR for Invasive plants by {results.get('baseline_fnr', 0) - results.get('critical_fnr', 0):.4f}.\n\n"
#                 else:
#                     report_str += "Critical Review did not reduce  binary FNR for Invasive plants.\n\n"
                
#                 # --- Critical Review Stats Table ---
#                 stats = results.get('critical_review_stats', {})
#                 if stats.get('total_reviews', 0) > 0:
#                     stats_table = [
#                         ["Metric", "Count"],
#                         ["Total Critical Reviews", stats.get('total_reviews', 0)],
#                         ["Correct -> Correct", stats.get('correct_to_correct', 0)],
#                         ["Correct -> Wrong (Harmful)", stats.get('correct_to_wrong', 0)],
#                         ["Wrong -> Correct (Helpful)", stats.get('wrong_to_correct', 0)],
#                         ["Wrong -> Wrong", stats.get('wrong_to_wrong', 0)],
#                         ["Success Rate", f"{stats.get('success_rate', 0) * 100:.2f}%"],
#                         ["Effectiveness", f"{stats.get('effectiveness', 0) * 100:.2f}%"]
#                     ]
#                     report_str += "Critical Review Breakdown:\n"
#                     report_str += tabulate(stats_table, headers="firstrow", tablefmt="grid")
#                     report_str += "\n"
                
#                 report_str += "-"*60 + "\n\n" 
            
#             page_title = f"A/B Summary: {model_desc}"
#             pdf.savefig(self._create_text_page(report_str, page_title))
#             plt.close()

#     def _add_detailed_metrics_pages(self, pdf, model_name, data_name, eval_pair):
#         """
#         Adds detailed metric pages for a given model and dataset, including
#         confusion matrices and formatted metric tables.
#         """
#         title = f"Detailed Metrics: {model_name} on {data_name}"
#         pdf.savefig(self._create_text_page(f"Detailed breakdown for:\n\nModel: {model_name}\nDataset: {data_name}", title))
#         plt.close()
        
#         for eval_type, evaluator in eval_pair.items():
#             # --- Page 1: Class-wise Confusion Matrix & Metrics Table ---
            
#             # Get the figure and the data for the metrics table
#             _, cm_fig = evaluator.get_classwise_confusion_matrix(display=False, return_fig=True)
#             classwise_results = evaluator.get_classwise_metrics(display=False)
            
#             # Format the tables into a string
#             summary_table_str = tabulate(classwise_results['summary_data'], headers=classwise_results['summary_headers'], tablefmt="grid")
#             classwise_table_str = tabulate(classwise_results['classwise_data'], headers=classwise_results['classwise_headers'], tablefmt="grid")
            
#             # Create a text page for the metrics
#             metrics_text = f"Overall Summary Metrics:\n{summary_table_str}\n\nPer-Class Metrics:\n{classwise_table_str}"
#             page_title = f"{eval_type.title()} - Class-wise Metrics"
            
#             # Add the pages to the PDF
#             cm_fig.suptitle(page_title) # Add a title to the figure
#             pdf.savefig(cm_fig)
#             plt.close(cm_fig)
#             pdf.savefig(self._create_text_page(metrics_text, page_title))
#             plt.close()

#             # --- Page 2: Binary Confusion Matrix & Metrics Table ---
            
#             # Get the figure and the data for the metrics table
#             _, bin_cm_fig = evaluator.get_binary_confusion_matrix(display=False, return_fig=True)
#             binary_metrics_dict = evaluator.get_binary_metrics(display=False)
            
#             # Format the dictionary into a two-column table string
#             binary_table_data = list(binary_metrics_dict.items())
#             binary_table_str = tabulate(binary_table_data, headers=["Metric", "Value"], tablefmt="grid")
            
#             # Create a text page for the metrics
#             binary_metrics_text = f"Binary Classification Metrics (Invasive vs. Non-Invasive):\n\n{binary_table_str}"
#             page_title = f"{eval_type.title()} - Binary Metrics"
            
#             # Add the pages to the PDF
#             bin_cm_fig.suptitle(page_title) # Add a title to the figure
#             pdf.savefig(bin_cm_fig)
#             plt.close(bin_cm_fig)
#             pdf.savefig(self._create_text_page(binary_metrics_text, page_title))
#             plt.close()

#     def _add_per_sample_pages(self, pdf, predictor, dataloader):
#         title = f"Per-Sample Analysis: {predictor.model_name}"
#         pdf.savefig(self._create_text_page(f"Detailed analysis for individual samples, using Model A ({predictor.model_name}).", title)); plt.close()
#         for i, (image, label) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Generating sample pages"):
#             pred_info = predictor.predict(input=image, true_label=label, return_fig=True, display=False, k=5, sample_id=i + 1)
#             if "figure" in pred_info:
#                 pdf.savefig(pred_info["figure"]); plt.close(pred_info["figure"])
#             if pred_info.get("is_critical") and "critical_review_figure" in pred_info:
#                 pdf.savefig(pred_info["critical_review_figure"]); plt.close(pred_info["critical_review_figure"])

#     @staticmethod
#     def _create_text_page(text_content: str, title: str):
#         fig = plt.figure(figsize=cfg.PDF_PAGE_SIZE)
#         fig.clf()
#         fig.suptitle(title, fontsize=16)
#         fig.text(0.05, 0.9, text_content, transform=fig.transFigure, size=10, ha="left", va="top", fontfamily="monospace")
#         return fig
