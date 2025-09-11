# src/experiment_manager.py
import json
import yaml
import shutil
from pathlib import Path
from datetime import datetime, timedelta
import torch
import pandas as pd
from sklearn.metrics import classification_report, jaccard_score
import numpy as np

# Import plotter từ cùng package src
from . import plotter

class ExperimentManager:
    def __init__(self, run_path: str, class_names: list, is_new_run: bool = True):
        self.output_dir = Path(run_path)
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.models_dir = self.output_dir / "models"
        self.history_path = self.output_dir / "history.csv"
        self.samples_dir = self.output_dir / "prediction_samples"
        if is_new_run:
            try:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                self.models_dir.mkdir(exist_ok=True)
            except OSError as e:
                print(f"FATAL: Could not create directory {self.output_dir}. Error: {e}")
                raise
        print(f"ExperimentManager initialized. Path: '{self.output_dir}'")
        self.start_time = datetime.now()
        self._load_history()

    @staticmethod
    def setup_run_directory(base_output_dir, run_name, resume_path=None):
        if resume_path:
            run_dir = Path(resume_path)
            if not run_dir.is_dir():
                raise FileNotFoundError(f"Resume path does not exist: {run_dir}")
            print(f"Resuming from specified path: {run_dir}")
            return str(run_dir), False
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_run_dir = Path(base_output_dir) / f"{run_name}_{timestamp}"
        return str(final_run_dir), True

    def _load_history(self):
        try:
            if self.history_path.exists():
                self.history_df = pd.read_csv(self.history_path)
            else:
                self.history_df = pd.DataFrame()
        except pd.errors.EmptyDataError:
            print(f"Warning: history.csv at {self.history_path} is empty. Starting fresh.")
            self.history_df = pd.DataFrame()
        except Exception as e:
            print(f"Error loading history.csv: {e}. Starting with an empty history.")
            self.history_df = pd.DataFrame()

    def save_config(self, config: dict):
        config_path = self.output_dir / "config.yaml"
        try:
            with open(config_path, 'w') as f:
                yaml.dump(config, f, indent=4, sort_keys=False)
            print(f"✓ Final configuration saved to {config_path}")
        except Exception as e:
            print(f"Error saving config file: {e}")

    def log_epoch(self, metrics: dict):
        is_new_file = not self.history_path.exists() or (self.history_df is not None and self.history_df.empty)
        new_row = pd.DataFrame([metrics])
        try:
            new_row.to_csv(self.history_path, mode='a', header=is_new_file, index=False)
            self.history_df = pd.concat([self.history_df, new_row], ignore_index=True)
        except Exception as e:
            print(f"Error writing to history.csv: {e}")

    def save_checkpoint(self, state: dict, is_best: bool):
        try:
            torch.save(state, self.models_dir / "latest_checkpoint.pth")
            if is_best:
                torch.save(state['model_state_dict'], self.models_dir / "best_model.pth")
        except Exception as e:
            print(f"Error saving checkpoint: {e}")

    def save_prediction_samples(self, samples: list):
        self.samples_dir.mkdir(exist_ok=True)
        try:
            # Xóa các sample cũ trước khi lưu sample mới
            for old_sample in self.samples_dir.glob("*.npz"):
                old_sample.unlink()
            
            for i, (img, gt, pred) in enumerate(samples):
                np.savez_compressed(
                    self.samples_dir / f"sample_{i+1}.npz",
                    image=img, ground_truth=gt, prediction=pred
                )
        except Exception as e:
            print(f"Error saving prediction samples: {e}")

    def load_checkpoint(self, filename="latest_checkpoint.pth"):
        checkpoint_path = self.models_dir / filename
        if not checkpoint_path.exists():
            return None
        try:
            return torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        except Exception as e:
            print(f"Error loading checkpoint {filename}: {e}")
            return None

    def checkpoint_exists(self, filename="latest_checkpoint.pth"):
        return (self.models_dir / filename).exists()

    def generate_final_report(self, best_miou: float, true_labels, pred_labels, config: dict, performance: dict):
        end_time = datetime.now()
        training_duration = self.start_time - end_time # Should be end_time - start_time
        training_duration = end_time - self.start_time
        
        try:
            per_class_metrics = self._calculate_per_class_metrics(true_labels, pred_labels)
            
            final_metrics = {
                'best_miou_from_training': best_miou,
                'performance': performance,
                'per_class_metrics_from_final_eval': per_class_metrics,
                'classification_report_from_final_eval': classification_report(true_labels, pred_labels, target_names=self.class_names, zero_division=0, output_dict=True)
            }
            with open(self.output_dir / 'final_metrics.json', 'w') as f:
                json.dump(final_metrics, f, indent=4)

            self._write_classification_report(best_miou, training_duration, per_class_metrics, config, performance)
            
            plotter.plot_training_history(self.history_df, self.output_dir / "training_metrics.png")
            plotter.plot_confusion_matrix(true_labels, pred_labels, self.class_names, self.output_dir / "confusion_matrix.png")
            plotter.plot_per_class_metrics(per_class_metrics, self.output_dir / "per_class_metrics.png")
        except Exception as e:
            print(f"An error occurred during final report generation: {e}")

    def _calculate_per_class_metrics(self, y_true, y_pred):
        iou_scores = jaccard_score(y_true, y_pred, average=None, labels=range(self.num_classes), zero_division=0)
        dice_scores = []
        for i in range(self.num_classes):
            yt = (y_true == i); yp = (y_pred == i)
            intersection = np.sum(yt & yp); union = np.sum(yt) + np.sum(yp)
            dice = (2. * intersection) / (union + 1e-8)
            dice_scores.append(dice)
        return {name: {'iou': iou, 'dice': dice} for name, iou, dice in zip(self.class_names, iou_scores, dice_scores)}

    def _write_classification_report(self, best_miou, duration, per_class_metrics, config, performance):
        with open(self.output_dir / 'final_metrics.json', 'r') as f:
            final_metrics = json.load(f)
        report_dict = final_metrics['classification_report_from_final_eval']
        report_df = pd.DataFrame(report_dict).transpose()
        report_str = report_df.to_string(float_format="%.4f")

        full_report_str = f"--- Final Training Report ---\n\n"
        full_report_str += f"Run Name: {config['run_name']}\n"
        full_report_str += f"Full Path: {self.output_dir}\n"
        full_report_str += f"Training Duration: {str(duration).split('.')[0]}\n"
        full_report_str += f"Completed Epochs: {len(self.history_df)}\n"
        full_report_str += f"Best Validation mIoU: {best_miou:.4f}\n\n"
        
        full_report_str += "--- Performance Benchmark ---\n"
        fps = performance.get('fps', 0)
        latency = performance.get('latency_ms', float('inf'))
        adas_latency_req_ms = 33.3
        adas_met = "YES" if latency <= adas_latency_req_ms else "NO"
        full_report_str += f"- FPS: {fps:.2f}\n"
        full_report_str += f"- Latency: {latency:.2f} ms\n"
        full_report_str += f"- ADAS Latency Requirement (<= {adas_latency_req_ms:.1f} ms): {adas_met}\n\n"

        full_report_str += "--- Per-Class Metrics (from final evaluation) ---\n"
        for class_name, metrics in per_class_metrics.items():
            full_report_str += f"- {class_name:<25}: IoU = {metrics['iou']:.4f}, Dice = {metrics['dice']:.4f}\n"
        
        full_report_str += "\n--- Classification Report (Scikit-learn) ---\n"
        full_report_str += report_str
        
        print("\n" + full_report_str)
        with open(self.output_dir / "classification_report.txt", "w") as f:
            f.write(full_report_str)

    @staticmethod
    def analyze_run(run_path: str):
        """
        Analyzes a training run robustly, generating all possible reports and plots
        from the available data (history.csv, final_metrics.json, samples).
        """
        run_path = Path(run_path)
        print(f"\n--- Analyzing Run: {run_path.name} ---")
        
        config_path = run_path / 'config.yaml'
        history_path = run_path / 'history.csv'
        
        if not config_path.exists() or not history_path.exists():
            print(f"Error: Missing required files for analysis.")
            if not config_path.exists(): print(f"- Required: {config_path}")
            if not history_path.exists(): print(f"- Required: {history_path}")
            return

        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            history_df = pd.read_csv(history_path)
        except (yaml.YAMLError, pd.errors.EmptyDataError) as e:
            print(f"Error: A data file seems corrupted or empty: {e}")
            return

        if history_df.empty:
            print("Analysis skipped: history.csv is empty.")
            return

        class_names = config.get('class_names', [])
        print("✓ Loaded config and history successfully.")
        
        print("\n--- Generating Plots from Training History ---")
        plotter.plot_training_history(history_df, run_path / "training_metrics_re-plot.png")
        
        best_epoch_idx = history_df['val_miou'].idxmax()
        best_epoch_stats = history_df.loc[best_epoch_idx]
        
        print("\n--- Summary from history.csv (Best Validation Epoch) ---")
        print(f"Best performance found at Epoch: {int(best_epoch_stats['epoch'])}")
        print(f"  - Best Validation mIoU: {best_epoch_stats['val_miou']:.4f}")
        print(f"  - Validation Accuracy:  {best_epoch_stats['val_acc']:.4f}")
        print(f"  - Validation Loss:      {best_epoch_stats['val_loss']:.4f}")
        print(f"  - Training Loss:        {best_epoch_stats['train_loss']:.4f}")
        print("----------------------------------------------------")

        final_metrics_path = run_path / 'final_metrics.json'
        if final_metrics_path.exists():
            print("\n--- Analysis from final_metrics.json (Completed Run) ---")
            with open(final_metrics_path, 'r') as f:
                final_metrics = json.load(f)
            
            performance = final_metrics.get('performance', {})
            if performance:
                fps = performance.get('fps', 0)
                latency = performance.get('latency_ms', float('inf'))
                adas_met = "YES" if latency <= 33.3 else "NO"
                print("Performance Benchmark:")
                print(f"  - FPS: {fps:.2f}")
                print(f"  - Latency: {latency:.2f} ms")
                print(f"  - ADAS Latency Met: {adas_met}")

            per_class_metrics = final_metrics.get('per_class_metrics_from_final_eval', {})
            if per_class_metrics:
                plotter.plot_per_class_metrics(per_class_metrics, run_path / "per_class_metrics_re-plot.png")
                print("\nIoU per class (from final evaluation):")
                for name, metrics in per_class_metrics.items():
                    print(f"  - {name:<25}: {metrics.get('iou', 'N/A'):.4f}")
            print("---------------------------------------------------------")
        else:
            print("\nNote: 'final_metrics.json' not found. The run may have been stopped before final evaluation.")

        # --- FIX: Luôn tìm và vẽ các mẫu dự đoán nếu có ---
        samples_dir = run_path / "prediction_samples"
        if samples_dir.is_dir():
            print("\n--- Visualizing Best Prediction Samples ---")
            samples_to_plot = []
            sample_files = sorted(samples_dir.glob("*.npz"))
            if not sample_files:
                print("  - Directory 'prediction_samples' exists but contains no .npz files.")
            else:
                try:
                    for sample_file in sample_files:
                        data = np.load(sample_file)
                        samples_to_plot.append((data['image'], data['ground_truth'], data['prediction']))
                    
                    plotter.plot_segmentation_results(samples_to_plot, class_names, run_path / "visual_results")
                except Exception as e:
                    print(f"  - Could not generate visualization plots. Error: {e}")
        else:
            print("\nNote: No prediction samples found for visualization (directory does not exist).")
        # ----------------------------------------------------