# src/plotter.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from pathlib import Path
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

def smooth(scalars, weight):
    if not scalars: return []
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def plot_training_history(history_df: pd.DataFrame, output_path: str, smoothing_weight=0.6):
    if history_df.empty:
        print("Warning: History is empty. Skipping plot generation.")
        return
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('Training and Validation Metrics', fontsize=16)
    epochs = history_df['epoch']
    train_loss = history_df['train_loss']
    val_loss = history_df['val_loss']
    axes[0, 0].plot(epochs, train_loss, 'c--', alpha=0.5, label='Train Loss (raw)')
    axes[0, 0].plot(epochs, val_loss, 'm--', alpha=0.5, label='Validation Loss (raw)')
    axes[0, 0].plot(epochs, smooth(train_loss.tolist(), smoothing_weight), 'c', label=f'Train Loss (smoothed)')
    axes[0, 0].plot(epochs, smooth(val_loss.tolist(), smoothing_weight), 'm', label=f'Validation Loss (smoothed)')
    axes[0, 0].set_title('Loss over Epochs'); axes[0, 0].legend(); axes[0, 0].grid(True)
    axes[0, 1].plot(epochs, history_df['val_acc'], label='Validation Accuracy', color='green')
    axes[0, 1].set_title('Validation Accuracy'); axes[0, 1].legend(); axes[0, 1].grid(True)
    axes[1, 0].plot(epochs, history_df['val_miou'], label='Validation mIoU', color='red')
    axes[1, 0].set_title('Validation mIoU'); axes[1, 0].legend(); axes[1, 0].grid(True)
    axes[1, 1].plot(epochs, history_df['lr'], label='Learning Rate', color='purple')
    axes[1, 1].set_title('Learning Rate Schedule'); axes[1, 1].legend(); axes[1, 1].grid(True)
    for ax in axes.flat:
        ax.set_xlabel('Epoch')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path)
    plt.close(fig)
    print(f"✓ Metrics plot saved to {output_path}")

def plot_confusion_matrix(true_labels, pred_labels, class_names, output_path):
    cm = confusion_matrix(true_labels, pred_labels, labels=range(len(class_names)))
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix'); plt.xlabel('Predicted'); plt.ylabel('True')
    plt.savefig(output_path)
    plt.close()
    print(f"✓ Confusion matrix saved to {output_path}")

def plot_per_class_metrics(metrics: dict, output_path: str):
    class_names = list(metrics.keys())
    iou_scores = [m['iou'] for m in metrics.values()]
    dice_scores = [m['dice'] for m in metrics.values()]
    x = np.arange(len(class_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(14, 8))
    rects1 = ax.bar(x - width/2, iou_scores, width, label='IoU', color='skyblue')
    rects2 = ax.bar(x + width/2, dice_scores, width, label='Dice', color='salmon')
    ax.set_ylabel('Scores')
    ax.set_title('Per-Class IoU and Dice Scores')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.legend()
    ax.bar_label(rects1, padding=3, fmt='%.3f')
    ax.bar_label(rects2, padding=3, fmt='%.3f')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    fig.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    print(f"✓ Per-class metrics plot saved to {output_path}")

def mask_to_rgb(mask, palette):
    """Converts a segmentation mask to an RGB image using a fixed palette."""
    rgb_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_idx, color in enumerate(palette):
        rgb_mask[mask == class_idx] = color
    return rgb_mask

def plot_segmentation_results(samples: list, class_names: list, output_dir: Path):
    """
    Plots a comparison of Original Image, Ground Truth, and Model Prediction
    with synchronized colors for all classes.
    """
    output_dir.mkdir(exist_ok=True)
    num_classes = len(class_names)
    
    # --- FIX: Create a fixed, visually distinct color palette ---
    # Lấy N màu từ một colormap chất lượng cao (ví dụ: tab10, tab20)
    # Chuyển đổi sang định dạng RGB 0-255
    palette = (plt.get_cmap('viridis', num_classes)(np.arange(num_classes))[:, :3] * 255).astype(np.uint8)
    # ---------------------------------------------------------

    for i, (image, gt, pred) in enumerate(samples):
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        
        # 1. Original Image
        if image.shape[2] > 3:
            rgb_image = image[:, :, [15, 8, 2]] 
            rgb_image = np.clip((rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min() + 1e-8), 0, 1)
        else:
            rgb_image = image
        axes[0].imshow(rgb_image)
        axes[0].set_title('Original Image (False Color)')
        axes[0].axis('off')

        # 2. Ground Truth (chuyển đổi sang RGB)
        gt_rgb = mask_to_rgb(gt, palette)
        axes[1].imshow(gt_rgb)
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')

        # 3. Prediction (chuyển đổi sang RGB)
        pred_rgb = mask_to_rgb(pred, palette)
        axes[2].imshow(pred_rgb)
        axes[2].set_title('Model Prediction')
        axes[2].axis('off')

        # Tạo legend chung
        legend_elements = [Patch(facecolor=color/255., label=name) for name, color in zip(class_names, palette)]
        fig.legend(handles=legend_elements, loc='lower center', ncol=num_classes, bbox_to_anchor=(0.5, -0.05))
        
        plt.tight_layout(rect=[0, 0.05, 1, 1]) # Điều chỉnh layout để legend không bị che
        save_path = output_dir / f"comparison_{i+1}.png"
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    print(f"✓ Saved {len(samples)} comparison plots to '{output_dir}'")