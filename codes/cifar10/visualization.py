"""
可视化工具

- 卷积核可视化
- 训练曲线 (loss & accuracy)
- 实验对比柱状图
- 混淆矩阵
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIGS_DIR = os.path.join(ROOT_DIR, 'results', 'task1')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results', 'task1')
os.makedirs(FIGS_DIR, exist_ok=True)

CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


def visualize_filters(model, layer_name='conv_blocks.0.conv', save_path=None):
    """可视化卷积核 (最多显示 32 个 filter 的前 3 通道)"""
    import torch

    target_layer = None
    for name, module in model.named_modules():
        if name == layer_name:
            target_layer = module
            break

    if target_layer is None:
        print(f"未找到层 '{layer_name}'. 可用层:")
        for name, _ in model.named_modules():
            if 'conv' in name.lower():
                print(f"  {name}")
        return

    weights = target_layer.weight.data.cpu().numpy()
    out_ch, in_ch, kh, kw = weights.shape

    n_cols = min(8, out_ch)
    n_rows = min(4, (out_ch + 7) // 8)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 1.5, n_rows * 1.5))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    show_ch = min(3, in_ch)
    for i in range(min(n_rows * n_cols, out_ch)):
        kernel = weights[i, :show_ch, :, :]
        kernel = (kernel - kernel.min()) / (kernel.max() - kernel.min() + 1e-8)
        if show_ch == 3:
            kernel = np.transpose(kernel, (1, 2, 0))
        elif show_ch == 1:
            kernel = kernel[0]
        elif show_ch == 2:
            pad = np.zeros((1, kh, kw))
            kernel = np.transpose(np.concatenate([kernel, pad], axis=0), (1, 2, 0))
        axes[i].imshow(kernel, cmap='gray' if show_ch == 1 else None)
        axes[i].set_title(f'Ch {i}')
        axes[i].axis('off')

    for i in range(out_ch, len(axes)):
        axes[i].axis('off')

    fig.suptitle(f'Conv Filters: {layer_name}\n({out_ch} filters, {kh}x{kw})',
                 fontsize=12)
    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(FIGS_DIR, 'conv_filters.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"卷积核可视化保存至 {save_path}")


def plot_training_curves(history, save_path=None, title='Training Curves'):
    """绘制训练/验证 loss 和 accuracy 曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history['train_loss']) + 1)

    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=1.5)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=1.5)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{title} - Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=1.5)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=1.5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'{title} - Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(FIGS_DIR, 'training_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"训练曲线保存至 {save_path}")


def plot_experiment_comparison(results, metric='test_acc', save_path=None, title=None):
    """绘制实验对比柱状图"""
    names = list(results.keys())
    values = [r[metric] * 100 if metric.endswith('acc') else r[metric]
              for r in results.values()]

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 0.8), 5))

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(names)))
    bars = ax.bar(range(len(names)), values, color=colors, edgecolor='black', linewidth=0.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f'{val:.2f}', ha='center', va='bottom', fontsize=8, rotation=45)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Accuracy (%)' if metric.endswith('acc') else metric)
    ax.set_title(title or f'Experiment Comparison ({metric})')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(FIGS_DIR, f'comparison_{metric}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"实验对比图保存至 {save_path}")


def plot_confusion_matrix(cm, save_path=None, normalize=False):
    """绘制混淆矩阵"""
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
           xlabel='Predicted', ylabel='True',
           title='Confusion Matrix')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black',
                    fontsize=8)

    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(FIGS_DIR, 'confusion_matrix.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def load_and_visualize_all():
    """加载所有实验结果, 生成可视化"""
    print("从实验结果生成可视化...")

    summary_path = os.path.join(RESULTS_DIR, 'summary.json')
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
        plot_experiment_comparison(summary, metric='test_acc',
                                   title='Test Accuracy Comparison')

    result_files = [f for f in os.listdir(RESULTS_DIR) if f.startswith('result_')]
    for rf in result_files:
        path = os.path.join(RESULTS_DIR, rf)
        with open(path) as f:
            result = json.load(f)

        exp_name = result['exp_name']
        history = result['history']
        save_path = os.path.join(FIGS_DIR, f'training_curves_{exp_name}.png')
        plot_training_curves(history, save_path=save_path,
                            title=f'Training Curves - {exp_name}')

    print("所有可视化已生成.")


def generate_best_model_viz(model_path=None):
    """用最佳 DeepCNN 模型 (ensemble_s123) 生成卷积核可视化和混淆矩阵。

    用法:
      python visualization.py --best
      或直接调用 generate_best_model_viz()
    """
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from models import CIFAR10_DeepCNN

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if model_path is None:
        model_path = os.path.join(ROOT_DIR, 'best_models', 'task1', 'ensemble_s123.pth')

    model = CIFAR10_DeepCNN(
        stage_channels=(64, 128, 256, 512), stage_depths=(2, 2, 3, 1),
        fc_units=(512, 256), dropout_p=0.1,
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device).eval()
    print(f'[可视化] 已加载模型: {os.path.basename(model_path)}')

    # ---- 1. 卷积核可视化 (Stage 0, 第一个 ResidualBlock 的第一个 Conv) ----
    visualize_filters(model, layer_name='stages.0.0.conv1',
                     save_path=os.path.join(FIGS_DIR, 'conv_filters.png'))

    # ---- 2. 混淆矩阵 ----
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2470, 0.2435, 0.2616])
    test_set = datasets.CIFAR10(
        root=os.path.join(ROOT_DIR, 'dataset'), train=False,
        transform=transforms.Compose([transforms.ToTensor(), normalize]))
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=0)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_labels.append(y.numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    cm = np.zeros((10, 10), dtype=np.int64)
    for t, p in zip(all_labels, all_preds):
        cm[t, p] += 1

    plot_confusion_matrix(cm, save_path=os.path.join(FIGS_DIR, 'confusion_matrix.png'))

    # 输出每类准确率供报告使用
    per_class = cm.diagonal() / cm.sum(axis=1)
    print('每类准确率:')
    for i, (name, acc) in enumerate(zip(CLASS_NAMES, per_class)):
        print(f'  {name:12s}: {acc:.4f} ({acc*100:.1f}%)')
    print(f'总准确率: {cm.diagonal().sum() / cm.sum():.4f}')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--best':
        generate_best_model_viz()
    else:
        load_and_visualize_all()
