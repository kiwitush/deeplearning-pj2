"""
Task 2: VGG-A with and without Batch Normalization
        + Loss Landscape Analysis.

This script:
  1. Trains standard VGG-A (no BN)
  2. Trains VGG-A with BatchNorm
  3. Compares training results
  4. Analyzes loss landscape by training with multiple learning rates
     and measuring loss variation (Lipschitzness)

Run from codes/VGG_BatchNorm/ directory:
  python VGG_Loss_Landscape.py
"""

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import random
import platform
import multiprocessing

from models.vgg import VGG_A, VGG_A_BatchNorm, get_number_of_parameters
from data.loaders import get_cifar_loader


# ============================================================
# Configuration
# ============================================================
NUM_WORKERS = 0 if platform.system() == 'Windows' else 4
BATCH_SIZE = 128

# Paths
HOME_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIGURES_PATH = os.path.join(PROJECT_ROOT, 'results', 'task2')
MODELS_PATH = os.path.join(PROJECT_ROOT, 'best_models', 'task2')
os.makedirs(FIGURES_PATH, exist_ok=True)
os.makedirs(MODELS_PATH, exist_ok=True)

# Device
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Utilities
# ============================================================
def set_random_seeds(seed_value=2020, device='cpu'):
    """Set random seeds for reproducibility."""
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != 'cpu':
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@torch.no_grad()
def get_accuracy(model, loader, device):
    """Calculate classification accuracy."""
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return correct / total


# ============================================================
# Training
# ============================================================
def train(model, optimizer, criterion, train_loader, val_loader,
          scheduler=None, epochs_n=20, best_model_path=None):
    """Training loop that records per-step loss for landscape analysis.

    Returns:
        losses_list: list of per-batch losses (list of lists)
        learning_curve: list of per-epoch average training loss
        val_accuracy_curve: list of per-epoch validation accuracy
    """
    model.to(device)
    learning_curve = [np.nan] * epochs_n
    train_accuracy_curve = [np.nan] * epochs_n
    val_accuracy_curve = [np.nan] * epochs_n
    max_val_accuracy = 0
    max_val_accuracy_epoch = 0

    batches_n = len(train_loader)
    losses_list = []

    for epoch in range(epochs_n):
        if scheduler is not None:
            scheduler.step()

        model.train()
        loss_list = []
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for data in train_loader:
            x, y = data
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, y)
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            running_loss += loss.item() * x.size(0)
            pred = prediction.argmax(dim=1)
            running_correct += (pred == y).sum().item()
            running_total += x.size(0)

        losses_list.append(loss_list)
        learning_curve[epoch] = running_loss / running_total
        train_accuracy_curve[epoch] = running_correct / running_total

        # Validation
        val_acc = get_accuracy(model, val_loader, device)
        val_accuracy_curve[epoch] = val_acc

        # Save best model
        if val_acc > max_val_accuracy:
            max_val_accuracy = val_acc
            max_val_accuracy_epoch = epoch
            if best_model_path:
                torch.save(model.state_dict(), best_model_path)

        print(f"Epoch {epoch+1:3d}/{epochs_n} | "
              f"train loss: {learning_curve[epoch]:.4f} "
              f"train acc: {train_accuracy_curve[epoch]:.4f} | "
              f"val acc: {val_acc:.4f}", flush=True)

    print(f"Best val acc: {max_val_accuracy:.4f} at epoch {max_val_accuracy_epoch+1}")
    return losses_list, learning_curve, val_accuracy_curve


# ============================================================
# Training comparison: VGG_A vs VGG_A_BatchNorm
# ============================================================
def train_and_compare(epochs=20):
    """Train both VGG_A and VGG_A_BatchNorm and compare."""
    train_loader = get_cifar_loader(root='../../dataset/', batch_size=BATCH_SIZE,
                                    train=True, shuffle=True,
                                    num_workers=NUM_WORKERS)
    val_loader = get_cifar_loader(root='../../dataset/', batch_size=BATCH_SIZE,
                                  train=False, shuffle=False,
                                  num_workers=NUM_WORKERS)

    results = {}

    for model_name, model_cls in [('VGG_A', VGG_A), ('VGG_A_BatchNorm', VGG_A_BatchNorm)]:
        print(f"\n{'='*60}")
        print(f"Training {model_name}")
        print(f"{'='*60}")

        set_random_seeds(seed_value=2020, device=device)
        model = model_cls()
        n_params = get_number_of_parameters(model)
        print(f"Model: {model_name}, Params: {n_params:,}")

        lr = 0.001
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        model_path = os.path.join(MODELS_PATH, f'{model_name}.pth')

        losses_list, learning_curve, val_curve = train(
            model, optimizer, criterion, train_loader, val_loader,
            epochs_n=epochs, best_model_path=model_path,
        )

        results[model_name] = {
            'losses_list': losses_list,
            'learning_curve': learning_curve,
            'val_accuracy_curve': val_curve,
            'n_params': n_params,
        }

    # ---- Plot comparison ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    ax = axes[0]
    for name, res in results.items():
        label = 'VGG_A + BatchNorm' if 'BatchNorm' in name else 'VGG_A'
        ax.plot(res['learning_curve'], label=label)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy curve
    ax = axes[1]
    for name, res in results.items():
        label = 'VGG_A + BatchNorm' if 'BatchNorm' in name else 'VGG_A'
        ax.plot(res['val_accuracy_curve'], label=label)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Accuracy')
    ax.set_title('Validation Accuracy Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('VGG-A with vs without Batch Normalization', fontsize=14)
    plt.tight_layout()
    save_path = os.path.join(FIGURES_PATH, 'vgg_bn_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nComparison plot saved to {save_path}")

    return results


# ============================================================
# Loss Landscape Analysis
# ============================================================
def train_with_lr(model, lr, train_loader, epochs=5, save_path=None):
    """Train a model with a specific learning rate and return per-step losses.

    Saves the model after training if save_path is provided.
    """
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    all_losses = []

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            all_losses.append(loss.item())

    if save_path:
        torch.save(model.state_dict(), save_path)

    return all_losses


def analyze_loss_landscape(epochs=5):
    """Train models with different learning rates and compute loss landscape curves.

    Uses learning rates: [2e-3, 1e-3, 5e-4, 1e-4]

    For each step, computes max and min loss across all LR models,
    then plots fill_between to visualize loss landscape smoothness.

    Performed for BOTH VGG_A and VGG_A_BatchNorm.
    """
    learning_rates = [2e-3, 1e-3, 5e-4, 1e-4]
    print(f"\nLearning rates for landscape analysis: {learning_rates}")

    landscape_results = {}

    for model_name, model_cls in [('VGG_A', VGG_A), ('VGG_A_BatchNorm', VGG_A_BatchNorm)]:
        print(f"\n--- {model_name} Landscape Analysis ---")

        all_curves = {}
        for lr in learning_rates:
            print(f"  Training {model_name} with lr={lr}")
            set_random_seeds(seed_value=2020, device=device)
            # Create a fresh data loader for each run to avoid shuffle state issues
            train_loader = get_cifar_loader(root='../../dataset/', batch_size=BATCH_SIZE,
                                            train=True, shuffle=True,
                                            num_workers=NUM_WORKERS)
            model = model_cls()
            model_path = os.path.join(MODELS_PATH, f'{model_name}_lr{lr}.pth')
            losses = train_with_lr(model, lr, train_loader, epochs=epochs, save_path=model_path)
            all_curves[lr] = losses
            # Per-LR diagnostic
            final_avg = np.mean(losses[-50:]) if len(losses) >= 50 else np.mean(losses)
            print(f"    lr={lr}: final_loss(avg last 50)={final_avg:.4f}, "
                  f"min_loss={min(losses):.4f}, max_loss={max(losses):.4f}")

        # Compute max/min curves across all LRs
        n_steps = min(len(v) for v in all_curves.values())
        max_curve = []
        min_curve = []

        for step in range(n_steps):
            step_losses = [all_curves[lr][step] for lr in learning_rates]
            max_curve.append(max(step_losses))
            min_curve.append(min(step_losses))

        # Diagnostic
        gap_curve = [max_curve[i] - min_curve[i] for i in range(n_steps)]
        print(f"  {model_name} — n_steps={n_steps}, "
              f"max={max(max_curve):.4f}, min={min(min_curve):.4f}, "
              f"mean_gap={np.mean(gap_curve):.4f}, max_gap={max(gap_curve):.4f}")

        landscape_results[model_name] = {
            'all_curves': all_curves,
            'max_curve': max_curve,
            'min_curve': min_curve,
            'gap_curve': gap_curve,
            'learning_rates': learning_rates,
        }

    # ---- Plot: 2 subplots (raw + gap) ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    colors = {'VGG_A': ('#FF6B6B', '#FF0000'),  # red tones
              'VGG_A_BatchNorm': ('#4ECDC4', '#008080')}  # teal tones

    window = 50

    # Draw BN first, then VGG_A so the wider area is on top
    plot_order = ['VGG_A_BatchNorm', 'VGG_A']

    for model_name in plot_order:
        res = landscape_results[model_name]
        max_curve = res['max_curve']
        min_curve = res['min_curve']
        gap_curve = res['gap_curve']
        steps = range(len(max_curve))
        fill_color, line_color = colors[model_name]

        label_name = 'VGG_A + BatchNorm' if 'BatchNorm' in model_name else 'VGG_A'

        # Subplot 1: raw max/min landscape
        ax1.plot(steps, max_curve, color=line_color, linewidth=1, alpha=0.8)
        ax1.plot(steps, min_curve, color=line_color, linewidth=1, alpha=0.8,
                 linestyle='--')
        ax1.fill_between(steps, min_curve, max_curve, color=fill_color,
                         alpha=0.3, label=label_name)

        # Subplot 2: landscape gap (smoothed)
        gap_smooth = np.convolve(gap_curve, np.ones(window)/window, mode='valid')
        ax2.plot(gap_smooth, color=line_color, linewidth=1.5, alpha=0.9,
                 label=label_name)

    ax1.set_yscale('log')
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Loss landscape (log scale)')
    ax1.set_title('Loss Landscape: Max/Min Curves')
    ax1.legend(loc='upper right', fontsize=7)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Training Step (smoothed)')
    ax2.set_ylabel('Loss Gap (max - min)')
    ax2.set_title('Landscape Smoothness')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f'Loss Landscape Analysis (LRs: {learning_rates})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(FIGURES_PATH, 'loss_landscape.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nLoss landscape plot saved to {save_path}")

    # ---- Plot individual learning rate curves for each model ----
    for model_name in ['VGG_A', 'VGG_A_BatchNorm']:
        fig, ax = plt.subplots(figsize=(12, 5))
        res = landscape_results[model_name]
        title_name = 'VGG_A + BatchNorm' if 'BatchNorm' in model_name else 'VGG_A'
        for lr, curve in res['all_curves'].items():
            window = 50
            smoothed = np.convolve(curve, np.ones(window)/window, mode='valid')
            ax.plot(smoothed, linewidth=1, alpha=0.7, label=f'lr={lr}')

        ax.set_xlabel('Training Step')
        ax.set_ylabel('Loss')
        ax.set_title(f'{title_name} — Per-LR Training Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        save_path = os.path.join(FIGURES_PATH, f'landscape_{model_name}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    return landscape_results


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    multiprocessing.freeze_support()

    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30,
                        help='Training epochs for comparison')
    parser.add_argument('--landscape-epochs', type=int, default=30,
                        help='Training epochs for landscape analysis')
    parser.add_argument('--comparison', action='store_true',
                        help='Run VGG-A vs VGG-A+BN training comparison')
    parser.add_argument('--landscape', action='store_true',
                        help='Run loss landscape analysis')
    args = parser.parse_args()

    # If neither flag is specified, run both
    run_both = not args.comparison and not args.landscape

    print("=" * 60)
    print("Task 2: Batch Normalization")
    print("=" * 60)

    # Part 1: Train and compare VGG-A vs VGG-A+BatchNorm
    if run_both or args.comparison:
        comparison_results = train_and_compare(epochs=args.epochs)

    # Part 2: Loss landscape analysis
    if run_both or args.landscape:
        landscape_results = analyze_loss_landscape(epochs=args.landscape_epochs)

    print("\n" + "=" * 60)
    print("Task 2 complete!")
    print(f"Figures saved to {FIGURES_PATH}")
    print(f"Models saved to {MODELS_PATH}")
    print("=" * 60)
