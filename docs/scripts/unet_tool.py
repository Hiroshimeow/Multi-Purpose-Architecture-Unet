# -*- coding: utf-8 -*-
"""
unet_tool.py

A comprehensive toolkit class for handling logging, checkpointing, reporting,
and analysis for the U-Net training pipeline. This class automatically creates
a unique, timestamped directory for each run to prevent overwriting results.
"""

import json
from pathlib import Path
from datetime import datetime
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

class UnetTool:
    """
    A helper class to manage the lifecycle of a U-Net training run.

    Handles:
    - Creating a unique, timestamped directory for each run.
    - Saving training configuration.
    - Logging metrics per epoch.
    - Saving model checkpoints (latest and best).
    - Loading models for inference or resuming.
    - Generating final reports, including plots and classification metrics.
    """
    def __init__(self, base_output_dir: str, run_name: str, class_names: list):
        """
        Initializes the tool for a specific training run, creating a unique directory.

        Args:
            base_output_dir (str): The root directory for all training runs.
            run_name (str): A descriptive base name for the current run.
            class_names (list): A list of strings representing the class names
                                for reporting purposes.
        """
        self.base_output_dir = Path(base_output_dir)
        self.run_name = run_name
        self.class_names = class_names
        
        # --- Create a unique directory for this run ---
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_run_name = f"{self.run_name}_{timestamp}"
        self.output_dir = self.base_output_dir / unique_run_name
        # ---------------------------------------------

        self.models_dir = self.output_dir / "models"
        self.history = {
            'train_loss': [], 'val_loss': [], 'val_acc': [],
            'val_miou': [], 'lr': []
        }
        self.best_metric_value = 0.0

        # Create necessary directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        print(f"UnetTool initialized. Results for this run will be saved to:\n'{self.output_dir}'")

    def setup_run(self, config: dict):
        """
        Sets up the run by saving the configuration and printing system info.
        """
        config_path = self.output_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"✓ Configuration saved to {config_path}")

        print(f"--- System Information ---")
        print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        print("--------------------------")

    def log_epoch(self, metrics: dict):
        """
        Logs the metrics for a single epoch.
        """
        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(value)
        
        if 'val_miou' in metrics and metrics['val_miou'] > self.best_metric_value:
            self.best_metric_value = metrics['val_miou']

    def save_checkpoint(self, model, optimizer, epoch: int, is_best: bool):
        """
        Saves a training checkpoint.
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_miou': self.best_metric_value
        }
        
        latest_path = self.models_dir / "latest_checkpoint.pth"
        torch.save(checkpoint, latest_path)

        if is_best:
            best_path = self.models_dir / "best_model.pth"
            torch.save(model.state_dict(), best_path)

    def load_model_for_inference(self, model, device):
        """
        Loads the best model's weights for final evaluation or inference.
        """
        model_path = self.models_dir / "best_model.pth"
        if not model_path.exists():
            print(f"Warning: Best model file not found at {model_path}. Returning original model.")
            return model
        
        model.load_state_dict(torch.load(model_path, map_location=device))
        return model

    def generate_final_report(self, best_miou: float, true_labels, pred_labels):
        """
        Generates all final outputs: plots, confusion matrix, and text report.
        """
        print("   Generating training history plots...")
        self._plot_training_history()

        print("   Generating confusion matrix...")
        self._plot_confusion_matrix(true_labels, pred_labels)

        print("   Generating classification report...")
        self._write_classification_report(best_miou, true_labels, pred_labels)

    def _plot_training_history(self):
        """Internal method to plot and save training metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Training and Validation Metrics', fontsize=16)
        
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Loss over Epochs'); axes[0, 0].legend(); axes[0, 0].grid(True)
        
        axes[0, 1].plot(self.history['val_acc'], label='Validation Accuracy', color='green')
        axes[0, 1].set_title('Validation Accuracy'); axes[0, 1].legend(); axes[0, 1].grid(True)
        
        axes[1, 0].plot(self.history['val_miou'], label='Validation mIoU', color='red')
        axes[1, 0].set_title('Validation mIoU'); axes[1, 0].legend(); axes[1, 0].grid(True)
        
        axes[1, 1].plot(self.history['lr'], label='Learning Rate', color='purple')
        axes[1, 1].set_title('Learning Rate Schedule'); axes[1, 1].legend(); axes[1, 1].grid(True)
        
        for ax in axes.flat: ax.set_xlabel('Epoch')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        save_path = self.output_dir / "training_metrics.png"
        plt.savefig(save_path)
        plt.close(fig)
        print(f"✓ Metrics plot saved to {save_path}")

    def _plot_confusion_matrix(self, true_labels, pred_labels):
        """Internal method to plot and save the confusion matrix."""
        cm = confusion_matrix(true_labels, pred_labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix'); plt.xlabel('Predicted'); plt.ylabel('True')
        
        save_path = self.output_dir / "confusion_matrix.png"
        plt.savefig(save_path)
        plt.close()
        print(f"✓ Confusion matrix saved to {save_path}")

    def _write_classification_report(self, best_miou, true_labels, pred_labels):
        """Internal method to generate and save the classification report."""
        report = classification_report(true_labels, pred_labels, 
                                       target_names=self.class_names, zero_division=0)
        
        report_str = f"--- Final Classification Report ---\n\n"
        report_str += f"Run Name: {self.run_name}\n"
        report_str += f"Full Path: {self.output_dir}\n"
        report_str += f"Best Validation mIoU achieved: {best_miou:.4f}\n\n"
        report_str += report
        
        print("\nClassification Report:")
        print(report)
        
        save_path = self.output_dir / "classification_report.txt"
        with open(save_path, "w") as f:
            f.write(report_str)
        print(f"✓ Classification report saved to {save_path}")