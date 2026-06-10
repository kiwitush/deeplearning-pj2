"""
Task 1: CIFAR-10 训练工具

包含: 数据加载 / MixUp 增强 / 训练循环 / 早停 / 多优化器 / 多损失函数
"""

import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


def set_random_seeds(seed=42, device='cpu'):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device != 'cpu':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_cifar10_loaders(data_root='../../dataset', batch_size=128,
                        num_workers=2, use_augmentation=False):
    """CIFAR-10 数据加载: train(45000)/val(5000)/test(10000)"""
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2470, 0.2435, 0.2616])

    if use_augmentation:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    full_train = datasets.CIFAR10(root=data_root, train=True, download=True,
                                  transform=train_transform)
    test_set = datasets.CIFAR10(root=data_root, train=False, download=True,
                                transform=test_transform)

    train_size = 45000
    val_size = 5000
    train_set, val_set = torch.utils.data.random_split(
        full_train, [train_size, val_size],
        generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("CPU")
    return device


def mixup_data(x, y, alpha=0.2):
    """MixUp: 线性插值输入和标签对。

    返回 mixed_x, y_a, y_b, lam
      mixed_x = lam * x + (1-lam) * x_shuffled
      loss = lam * CE(logits, y_a) + (1-lam) * CE(logits, y_b)
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    lam = max(lam, 1 - lam)  # lam >= 0.5, 语义更清晰

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, logits, y_a, y_b, lam):
    """MixUp 损失: 加权两个标签对"""
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)


def rand_bbox(size, lam):
    """CutMix 随机矩形区域"""
    W, H = size, size
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, W)
    y2 = min(cy + cut_h // 2, H)
    return x1, y1, x2, y2


def cutmix_data(x, y, alpha=0.2):
    """CutMix: 将一张图的矩形区域替换为另一张图的对应区域。

    Returns mixed_x, y_a, y_b, lam
      lam = 保留 patch 面积占比
      loss = lam * CE(logits, y_a) + (1-lam) * CE(logits, y_b)
    """
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    x1, y1, x2, y2 = rand_bbox(x.size(-1), lam)
    lam = 1 - ((x2 - x1) * (y2 - y1)) / (x.size(-1) * x.size(-1))
    lam = max(lam, 1 - lam)

    mixed_x = x.clone()
    mixed_x[:, :, x1:x2, y1:y2] = x[index, :, x1:x2, y1:y2]
    return mixed_x, y, y[index], lam


def mix_augment(x, y, mixup_alpha=0.2, cutmix_alpha=0.2, cutmix_prob=0.3):
    """随机选择 MixUp 或 CutMix 数据增强。"""
    if np.random.rand() < cutmix_prob:
        return cutmix_data(x, y, cutmix_alpha)
    else:
        return mixup_data(x, y, mixup_alpha)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """评估模型: 返回 (avg_loss, accuracy)"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    model.train()
    return total_loss / total, correct / total


def train_epoch(model, loader, optimizer, criterion, device, mixup_alpha=0.0,
                cutmix_alpha=0.0, cutmix_prob=0.0):
    """单 epoch 训练, 返回 (avg_loss, accuracy)"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    do_augment = (mixup_alpha > 0) or (cutmix_alpha > 0)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if do_augment:
            x, y_a, y_b, lam = mix_augment(x, y, mixup_alpha, cutmix_alpha, cutmix_prob)
        optimizer.zero_grad()
        logits = model(x)
        if do_augment:
            loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
        else:
            loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return running_loss / total, correct / total


def train(model, train_loader, val_loader, epochs, optimizer, criterion,
          scheduler=None, device='cpu', save_path=None, patience=15,
          swanlab_run=None, mixup_alpha=0.0, cutmix_alpha=0.0, cutmix_prob=0.0,
          warmup_epochs=0):
    """完整训练循环, 带早停和可选 SwanLab 日志。

    Args:
        model: nn.Module
        train_loader, val_loader: DataLoader
        epochs: 最大 epoch 数
        optimizer, criterion: 优化器和损失函数
        scheduler: 可选 lr scheduler (支持 ReduceLROnPlateau)
        save_path: 最佳模型保存路径
        patience: 早停耐心值 (epochs)
        swanlab_run: SwanLab run 对象
        mixup_alpha: MixUp Beta 分布 alpha (0=关闭)
        cutmix_alpha: CutMix Beta 分布 alpha (0=关闭)
        cutmix_prob: CutMix vs MixUp 选择概率
        warmup_epochs: LR warmup epoch 数 (0=关闭)

    Returns:
        history: {train_loss, train_acc, val_loss, val_acc} 列表
        best_val_acc: 最佳验证准确率
        best_epoch: 最佳 epoch 编号
    """
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    best_epoch = 0
    no_improve = 0

    base_lr = optimizer.param_groups[0]['lr']
    for epoch in range(1, epochs + 1):
        t_start = time.time()

        # Warmup: 线性增长 lr
        if warmup_epochs > 0 and epoch <= warmup_epochs:
            lr = base_lr * epoch / warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = lr

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion,
                                             device, mixup_alpha, cutmix_alpha, cutmix_prob)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                # Warmup 期间不调用 scheduler, warmup 后再用
                if warmup_epochs == 0 or epoch > warmup_epochs:
                    scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            no_improve = 0
            if save_path:
                torch.save(model.state_dict(), save_path)
        else:
            no_improve += 1

        t_elapsed = time.time() - t_start

        if swanlab_run is not None:
            lr = optimizer.param_groups[0]['lr']
            swanlab_run.log({
                'train/loss': train_loss,
                'train/acc': train_acc,
                'val/loss': val_loss,
                'val/acc': val_acc,
                'train/lr': lr,
                'epoch': epoch,
            })

        print(f"Epoch {epoch:3d}/{epochs} | "
              f"train loss: {train_loss:.4f} acc: {train_acc:.4f} | "
              f"val loss: {val_loss:.4f} acc: {val_acc:.4f} | "
              f"time: {t_elapsed:.1f}s", flush=True)

        if no_improve >= patience:
            print(f"早停触发 at epoch {epoch}")
            break

    print(f"最佳 val acc: {best_val_acc:.4f} at epoch {best_epoch}")
    return history, best_val_acc, best_epoch


def get_optimizer(name, model, lr, weight_decay=0, momentum=0.9):
    """根据名称获取优化器"""
    name = name.lower()
    if name == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                         weight_decay=weight_decay, nesterov=True)
    elif name == 'adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"未知优化器: {name}")


class FocalLoss(nn.Module):
    """Focal Loss: 降权易分样本, 聚焦难分样本.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    gamma=0 时退化为普通 CE. gamma 越大, 易分样本权重越低.
    """

    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma
        if self.alpha is not None:
            if isinstance(self.alpha, (list, tuple)):
                alpha_t = torch.tensor(self.alpha, device=logits.device)[targets]
            else:
                alpha_t = self.alpha
            focal_weight = focal_weight * alpha_t
        loss = focal_weight * ce_loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


def get_criterion(name='ce', model=None, l1_lambda=0, l2_lambda=0,
                  label_smoothing=0.0, focal_gamma=2.0):
    """获取损失函数, 支持 L1/L2 正则化和 Label Smoothing。

    Args:
        name: 'ce' (CrossEntropy)
        model: 正则化时需要 model.parameters()
        l1_lambda: L1 正则化系数
        l2_lambda: L2 正则化系数
        label_smoothing: 标签平滑因子 (0=关闭)
    """
    if name == 'focal':
        base_criterion = FocalLoss(gamma=focal_gamma)
    else:
        base_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    if l1_lambda == 0 and l2_lambda == 0:
        return base_criterion

    class RegularizedLoss:
        def __init__(self, base_criterion, model, l1_lambda, l2_lambda):
            self.base = base_criterion
            self.model = model
            self.l1 = l1_lambda
            self.l2 = l2_lambda

        def __call__(self, logits, targets):
            loss = self.base(logits, targets)
            if self.l1 > 0:
                l1_reg = sum(p.abs().sum() for p in self.model.parameters())
                loss = loss + self.l1 * l1_reg
            if self.l2 > 0:
                l2_reg = sum(p.pow(2).sum() for p in self.model.parameters())
                loss = loss + self.l2 * l2_reg
            return loss

        def to(self, device):
            self.base.to(device)
            return self

    return RegularizedLoss(base_criterion, model, l1_lambda, l2_lambda)
