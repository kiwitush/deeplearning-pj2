"""
Test script for VGG-A with and without Batch Normalization.

Evaluates saved models on the CIFAR-10 test set.

Usage (from codes/VGG_BatchNorm/ directory):
  python test_vgg.py --model VGG_A
  python test_vgg.py --model VGG_A_BatchNorm
"""

import os
import argparse
import platform
import numpy as np
import torch
import torch.nn as nn

from models.vgg import VGG_A, VGG_A_BatchNorm, get_number_of_parameters
from data.loaders import get_cifar_loader

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
MODELS_PATH = os.path.join(PROJECT_ROOT, 'best_models', 'task2')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    class_correct = [0] * 10
    class_total = [0] * 10

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)
        for i in range(len(y)):
            label = y[i].item()
            class_correct[label] += (pred[i] == label).item()
            class_total[label] += 1

    acc = correct / total
    print(f"\nOverall Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print("\nPer-class Accuracy:")
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    for i in range(10):
        cls_acc = class_correct[i] / class_total[i]
        print(f"  {classes[i]:12s}: {cls_acc:.4f} ({cls_acc*100:.1f}%)")

    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='VGG_A',
                        choices=['VGG_A', 'VGG_A_BatchNorm', 'VGG_A_Light', 'VGG_A_Dropout'],
                        help='Model to test')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to saved model weights')
    args = parser.parse_args()

    print(f"Testing {args.model} on CIFAR-10 test set")
    print(f"Device: {device}")

    # Load model
    model_cls = {
        'VGG_A': VGG_A,
        'VGG_A_BatchNorm': VGG_A_BatchNorm,
        'VGG_A_Light': None,
        'VGG_A_Dropout': None,
    }
    cls = model_cls.get(args.model)
    if cls is None:
        # Fallback: import from models.vgg
        import importlib
        vgg = importlib.import_module('models.vgg')
        cls = getattr(vgg, args.model)

    model = cls()
    n_params = get_number_of_parameters(model)
    print(f"Parameters: {n_params:,}")

    # Load weights if provided
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Loaded weights from {args.model_path}")
    else:
        default_path = os.path.join(MODELS_PATH, f'{args.model}.pth')
        if os.path.exists(default_path):
            model.load_state_dict(torch.load(default_path, map_location=device))
            print(f"Loaded weights from {default_path}")
        else:
            print(f"Warning: No weights found at {default_path}. Testing untrained model.")

    # Prepare data
    test_loader = get_cifar_loader(
        root='../../dataset/', batch_size=args.batch_size,
        train=False, shuffle=False,
        num_workers=0 if platform.system() == 'Windows' else 2,
    )

    evaluate(model, test_loader)


if __name__ == '__main__':
    main()
