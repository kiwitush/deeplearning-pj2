"""
t-SNE 特征可视化 — DeepCNN Baseline

对 CIFAR-10 测试集提取 FC 层 256 维特征，t-SNE 降维到 2D 并可视化。

用法 (从 codes/cifar10/ 目录运行):
  python tsne_visualization.py
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import CIFAR10_DeepCNN

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(ROOT_DIR, 'best_models', 'task1', 'ensemble_s42.pth')
FIGS_DIR = os.path.join(ROOT_DIR, 'results', 'task1')
os.makedirs(FIGS_DIR, exist_ok=True)

CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
CLASS_COLORS = plt.cm.tab10(np.linspace(0, 1, 10))


def load_model(device):
    model = CIFAR10_DeepCNN(
        stage_channels=(64, 128, 256, 512),
        stage_depths=(2, 2, 3, 1),
        fc_units=(512, 256),
        activation='relu',
    )
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model = model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model params: {n_params:,}')
    return model


def extract_features(model, device, n_samples=3000):
    """提取测试集 256 维 FC 特征，随机采样 n_samples 个样本用于 t-SNE。"""
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2470, 0.2435, 0.2616])
    test_set = datasets.CIFAR10(
        root=os.path.join(ROOT_DIR, 'dataset'), train=False,
        transform=transforms.Compose([transforms.ToTensor(), normalize]))
    loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=0)

    all_feats, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            # 手动前向: stages → GAP → flatten → classifier[:-1] (256-dim)
            feat = x
            for stage in model.stages:
                feat = stage(feat)
            feat = model.gap(feat)
            feat = feat.view(feat.size(0), -1)
            feat = model.classifier[:-1](feat)
            all_feats.append(feat.cpu().numpy())
            all_labels.append(y.numpy())

    all_feats = np.concatenate(all_feats, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # 随机采样
    rng = np.random.RandomState(42)
    indices = rng.choice(len(all_feats), min(n_samples, len(all_feats)), replace=False)
    return all_feats[indices], all_labels[indices]


def run_tsne(features, perplexity=30, n_iter=1000):
    print(f'Running t-SNE on {features.shape[0]} samples, {features.shape[1]}-dim features...')
    tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=n_iter,
                random_state=42, verbose=1)
    embedded = tsne.fit_transform(features)
    return embedded


def plot_tsne(embedded, labels, save_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
        mask = labels == i
        ax.scatter(embedded[mask, 0], embedded[mask, 1],
                   c=[color], label=name, s=3, alpha=0.6, rasterized=True)
    ax.legend(markerscale=4, fontsize=8, loc='lower left', ncol=2,
              framealpha=0.9, edgecolor='gray')
    ax.set_title('t-SNE Visualization of DeepCNN Features (CIFAR-10 Test Set)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f't-SNE figure saved to {save_path}')


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    model = load_model(device)
    features, labels = extract_features(model, device, n_samples=3000)
    embedded = run_tsne(features, perplexity=30, n_iter=1000)

    save_path = os.path.join(FIGS_DIR, 'tsne.png')
    plot_tsne(embedded, labels, save_path)


if __name__ == '__main__':
    main()
