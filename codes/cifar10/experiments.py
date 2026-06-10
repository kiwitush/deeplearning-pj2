"""
Task 1: CIFAR-10 实验运行器

灵活 CLI — 所有参数有默认值 (baseline), 任意参数可覆盖。
结果 / 模型 / 训练曲线分实验保存。

用法:
  python experiments.py                                    # baseline
  python experiments.py --name my_exp --optimizer adam --lr 0.001
  python experiments.py --activation gelu --fc-units 1024 512 256
  python experiments.py --no-bn --no-dropout --name no_regularization
  python experiments.py --model resnet --epochs 100
  python experiments.py --model deepcnn --optimizer adamw --lr 0.001
  python experiments.py --swanlab-project my_proj
  python experiments.py --run-all                          # 批量跑所有预定义实验
  python experiments.py --run-all --quick                  # 快速模式 (10 epochs)
"""

import os
import sys
import json
import argparse
import inspect
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import CIFAR10_CNN, CIFAR10_DeepCNN, CIFAR10_ResNet, count_parameters
from train import (set_random_seeds, get_cifar10_loaders, get_device,
                   get_optimizer, get_criterion, train, evaluate)

# 路径
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BEST_MODELS_DIR = os.path.join(ROOT_DIR, 'best_models', 'task1')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results', 'task1')
os.makedirs(BEST_MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# 默认 baseline 配置
BASELINE_DEFAULTS = {
    'model': 'cnn',
    'conv_channels': (32, 64, 128, 256, 512, 512, 512),
    'fc_units': (512, 256),
    'activation': 'relu',
    'use_bn': True,
    'use_dropout': True,
    'dropout_rate': 0.1,
    'use_gap': True,
    'num_blocks': (3, 3, 3),
    'base_channels': 16,
    'stage_channels': (64, 128, 256, 512),
    'stage_depths': (2, 2, 3, 1),
    'optimizer': 'sgd',
    'lr': 0.01,
    'weight_decay': 5e-4,
    'momentum': 0.9,
    'batch_size': 128,
    'epochs': 100,
    'patience': 15,
    'l1_lambda': 0.0,
    'l2_lambda': 0.0,
    'label_smoothing': 0.0,
    'augmentation': True,
    'seed': 42,
}


# CLI
def build_parser():
    p = argparse.ArgumentParser(
        description='CIFAR-10 实验运行器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  python experiments.py                                          # baseline
  python experiments.py --name adam_test --optimizer adam --lr 0.001
  python experiments.py --activation gelu --epochs 100
  python experiments.py --no-bn --no-dropout --name no_reg
  python experiments.py --model resnet --epochs 100
  python experiments.py --model deepcnn --optimizer adamw --lr 0.001
  python experiments.py --run-all                               # 所有预定义实验
  python experiments.py --run-all --quick                       # 快速批量 (10 epochs)
        ''')

    # 运行模式
    p.add_argument('--run-all', action='store_true',
                   help='批量运行所有预定义实验')
    p.add_argument('--quick', action='store_true',
                   help='快速模式 (10 epochs)')

    # 实验标识
    p.add_argument('--name', type=str, default=None,
                   help='实验名 (省略则自动生成)')

    # 模型架构
    p.add_argument('--model', type=str, default='cnn',
                   choices=['cnn', 'deepcnn', 'resnet'],
                   help='模型架构 (默认: cnn)')
    p.add_argument('--conv-channels', type=int, nargs='+',
                   default=BASELINE_DEFAULTS['conv_channels'],
                   help='每层 conv 输出通道 (默认: 32 64 128 256 512 512 512)')
    p.add_argument('--fc-units', type=int, nargs='+',
                   default=BASELINE_DEFAULTS['fc_units'],
                   help='FC 隐藏单元 (默认: 512 256)')
    p.add_argument('--activation', type=str, default='relu',
                   choices=['relu', 'leaky_relu', 'gelu'],
                   help='激活函数 (默认: relu)')
    p.add_argument('--no-bn', action='store_true', default=False,
                   help='关闭 Batch Normalization')
    p.add_argument('--no-dropout', action='store_true', default=False,
                   help='关闭 Dropout')
    p.add_argument('--dropout-rate', type=float, default=0.1,
                   help='Dropout 概率 (默认: 0.1)')
    p.add_argument('--no-gap', action='store_true', default=False,
                   help='关闭 GAP (使用 flatten + FC)')
    p.add_argument('--num-blocks', type=int, nargs='+',
                   default=BASELINE_DEFAULTS['num_blocks'],
                   help='ResNet 每 stage block 数 (默认: 3 3 3)')
    p.add_argument('--base-channels', type=int, default=16,
                   help='ResNet 基础通道数 (默认: 16)')
    p.add_argument('--stage-channels', type=int, nargs='+',
                   default=[64, 128, 256, 512],
                   help='DeepCNN stage 通道数 (默认: 64 128 256 512)')
    p.add_argument('--stage-depths', type=int, nargs='+',
                   default=[2, 2, 3, 1],
                   help='DeepCNN 每 stage 残差块数 (默认: 2 2 3 1)')

    # 优化器
    p.add_argument('--optimizer', type=str, default='sgd',
                   choices=['sgd', 'adam', 'adamw'],
                   help='优化器 (默认: sgd)')
    p.add_argument('--lr', type=float, default=0.01,
                   help='学习率 (默认: 0.01)')
    p.add_argument('--weight-decay', type=float, default=5e-4,
                   help='权重衰减 (默认: 5e-4)')
    p.add_argument('--momentum', type=float, default=0.9,
                   help='SGD 动量 (默认: 0.9)')

    # 训练配置
    p.add_argument('--epochs', type=int, default=100,
                   help='最大训练 epoch 数 (默认: 100)')
    p.add_argument('--batch-size', type=int, default=128,
                   help='批次大小 (默认: 128)')
    p.add_argument('--patience', type=int, default=15,
                   help='早停耐心值 (默认: 15)')

    # 损失 / 正则化
    p.add_argument('--l1-lambda', type=float, default=0.0,
                   help='L1 正则化强度 (默认: 0)')
    p.add_argument('--l2-lambda', type=float, default=0.0,
                   help='L2 正则化强度 (默认: 0)')
    p.add_argument('--label-smoothing', type=float, default=0.0,
                   help='标签平滑因子 (默认: 0.0)')
    p.add_argument('--criterion', type=str, default='ce',
                   choices=['ce', 'focal'],
                   help='损失函数 (默认: ce)')
    p.add_argument('--focal-gamma', type=float, default=2.0,
                   help='Focal Loss gamma 参数 (默认: 2.0)')

    # 数据增强
    p.add_argument('--no-augment', action='store_true', default=False,
                   help='关闭数据增强')
    p.add_argument('--mixup-alpha', type=float, default=0.0,
                   help='MixUp alpha (Beta 分布). 0=关闭. 建议: 0.2~0.4')
    p.add_argument('--cutmix-alpha', type=float, default=0.0,
                   help='CutMix alpha (Beta 分布). 0=关闭. 建议: 0.2')
    p.add_argument('--cutmix-prob', type=float, default=0.0,
                   help='CutMix 选择概率 (0~1, 其余用 MixUp). 建议: 0.3')
    p.add_argument('--warmup-epochs', type=int, default=0,
                   help='LR warmup epoch 数 (默认: 0=关闭, 建议: 5)')
    p.add_argument('--seed', type=int, default=42,
                   help='随机种子 (默认: 42)')

    # SwanLab
    p.add_argument('--swanlab-project', type=str, default=None,
                   help='SwanLab 项目名 (默认: 自动生成)')

    return p


def generate_exp_name(args):
    """根据非默认参数自动生成实验名"""
    parts = []

    if args.model != BASELINE_DEFAULTS['model']:
        parts.append(args.model)

    if args.activation != BASELINE_DEFAULTS['activation']:
        parts.append(f"act_{args.activation}")

    if args.optimizer != BASELINE_DEFAULTS['optimizer']:
        parts.append(f"opt_{args.optimizer}")

    if args.lr != BASELINE_DEFAULTS['lr']:
        parts.append(f"lr{args.lr}")

    if args.weight_decay != BASELINE_DEFAULTS['weight_decay']:
        parts.append(f"wd{args.weight_decay}")

    if args.no_bn:
        parts.append("no_bn")
    if args.no_dropout:
        parts.append("no_drop")
    if args.no_augment:
        parts.append("no_aug")
    if args.dropout_rate != BASELINE_DEFAULTS['dropout_rate']:
        parts.append(f"dp{args.dropout_rate}")
    if args.label_smoothing > 0:
        parts.append(f"ls{args.label_smoothing}")
    if args.mixup_alpha > 0:
        parts.append(f"mix{args.mixup_alpha}")

    if args.l1_lambda > 0:
        parts.append(f"l1_{args.l1_lambda}")
    if args.l2_lambda > 0:
        parts.append(f"l2_{args.l2_lambda}")

    if args.conv_channels != BASELINE_DEFAULTS['conv_channels']:
        parts.append(f"conv{'_'.join(str(c) for c in args.conv_channels)}")
    if args.fc_units != BASELINE_DEFAULTS['fc_units']:
        parts.append(f"fc{'_'.join(str(u) for u in args.fc_units)}")

    if not parts:
        return 'baseline'

    return '_'.join(parts)


def build_config(args):
    """CLI 参数转实验配置 dict"""
    if args.model == 'resnet':
        model_cls = CIFAR10_ResNet
    elif args.model == 'deepcnn':
        model_cls = CIFAR10_DeepCNN
    else:
        model_cls = CIFAR10_CNN

    config = {
        'model_cls': model_cls,
        'conv_channels': tuple(args.conv_channels),
        'fc_units': tuple(args.fc_units),
        'activation': args.activation,
        'use_bn': not args.no_bn,
        'use_dropout': not args.no_dropout,
        'dropout_p': args.dropout_rate,
        'use_gap': not args.no_gap,
        'num_blocks': tuple(args.num_blocks),
        'base_channels': args.base_channels,
        'stage_channels': tuple(args.stage_channels),
        'stage_depths': tuple(args.stage_depths),
        'optimizer': args.optimizer,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'momentum': args.momentum,
        'batch_size': args.batch_size,
        'patience': args.patience,
        'l1_lambda': args.l1_lambda,
        'l2_lambda': args.l2_lambda,
        'label_smoothing': args.label_smoothing,
        'criterion': args.criterion,
        'focal_gamma': args.focal_gamma,
        'mixup_alpha': args.mixup_alpha,
        'cutmix_alpha': args.cutmix_alpha,
        'cutmix_prob': args.cutmix_prob,
        'warmup_epochs': args.warmup_epochs,
        'augmentation': not args.no_augment,
        'seed': args.seed,
    }
    return config


# 单次实验运行
def run_experiment(config, exp_name, epochs=100, device='cpu',
                   swanlab_project=None):
    """运行单次实验, 保存模型和结果"""
    print(f"\n{'='*60}")
    print(f"实验: {exp_name}")
    print(f"配置:")
    for k, v in config.items():
        if k != 'model_cls':
            print(f"  {k}: {v}")
    print(f"{'='*60}")

    # SwanLab 初始化
    swanlab_run = None
    try:
        import swanlab
        if swanlab_project is None:
            swanlab_project = "cifar10"
        swanlab_run = swanlab.init(
            project=swanlab_project,
            experiment_name=exp_name,
            config=config,
        )
    except ImportError:
        print("[SwanLab] 未安装, 跳过监控")

    seed = config.get('seed', 42)
    set_random_seeds(seed, device.type if hasattr(device, 'type') else device)

    train_loader, val_loader, test_loader = get_cifar10_loaders(
        batch_size=config.get('batch_size', 128),
        use_augmentation=config.get('augmentation', True),
    )

    # 构建模型 (只传模型支持的参数)
    model_cls = config.get('model_cls', CIFAR10_CNN)
    valid_params = set(inspect.signature(model_cls.__init__).parameters.keys())
    valid_params.discard('self')
    model_kwargs = {k: v for k, v in config.items() if k in valid_params}

    model = model_cls(**model_kwargs).to(device)
    n_params = count_parameters(model)
    print(f"模型参数量: {n_params:,}")

    optimizer = get_optimizer(config.get('optimizer', 'sgd'), model,
                              lr=config.get('lr', 0.01),
                              weight_decay=config.get('weight_decay', 5e-4),
                              momentum=config.get('momentum', 0.9))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    criterion = get_criterion(name=config.get('criterion', 'ce'), model=model,
                              l1_lambda=config.get('l1_lambda', 0),
                              l2_lambda=config.get('l2_lambda', 0),
                              label_smoothing=config.get('label_smoothing', 0.0),
                              focal_gamma=config.get('focal_gamma', 2.0))

    save_path = os.path.join(BEST_MODELS_DIR, f'{exp_name}.pth')
    history, best_val_acc, best_epoch = train(
        model, train_loader, val_loader, epochs=epochs,
        optimizer=optimizer, criterion=criterion, scheduler=scheduler,
        device=device, save_path=save_path,
        patience=config.get('patience', 15),
        swanlab_run=swanlab_run,
        mixup_alpha=config.get('mixup_alpha', 0.0),
        cutmix_alpha=config.get('cutmix_alpha', 0.0),
        cutmix_prob=config.get('cutmix_prob', 0.0),
        warmup_epochs=config.get('warmup_epochs', 0),
    )

    # 加载最佳模型并在测试集上评估
    model.load_state_dict(torch.load(save_path, map_location=device))
    test_criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(model, test_loader, test_criterion, device)

    if swanlab_run is not None:
        swanlab_run.log({'test/loss': test_loss, 'test/acc': test_acc})
        swanlab_run.finish()

    result = {
        'exp_name': exp_name,
        'config': {k: str(v) for k, v in config.items()},
        'n_params': n_params,
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'test_loss': test_loss,
        'test_acc': test_acc,
        'history': {k: [float(x) for x in v] for k, v in history.items()},
    }

    result_path = os.path.join(RESULTS_DIR, f'result_{exp_name}.json')
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n{'='*60}")
    print(f"*** 测试结果: {test_acc:.4f} ({test_acc*100:.2f}%) ***")
    print(f"{'='*60}")
    print(f"模型保存至 {save_path}")
    print(f"结果保存至 {result_path}")
    return result


# 批量: 所有预定义实验
def run_all_experiments(device='cpu', quick=False):
    """运行全部预定义实验"""
    epochs = 10 if quick else 100
    results = {}

    baseline_config = {
        'model_cls': CIFAR10_CNN,
        'conv_channels': (32, 64, 128, 256, 512, 512, 512),
        'fc_units': (512, 256),
        'activation': 'relu',
        'use_bn': True,
        'use_dropout': True,
        'use_gap': True,
        'optimizer': 'sgd',
        'lr': 0.01,
        'weight_decay': 5e-4,
        'augmentation': True,
        'patience': 15,
    }

    results['baseline'] = run_experiment(baseline_config, 'baseline', epochs, device)

    # 不同滤波器数量
    filter_configs = {
        'narrow': (16, 32, 64, 128, 256, 256, 256),
        'wide': (64, 128, 256, 512, 512, 512, 512),
    }
    for name, channels in filter_configs.items():
        cfg = baseline_config.copy()
        cfg['conv_channels'] = channels
        results[f'filters_{name}'] = run_experiment(cfg, f'filters_{name}', epochs, device)

    # 不同 FC 大小
    fc_configs = {
        'small': (256, 128),
        'large': (1024, 512, 256),
    }
    for name, fc_units in fc_configs.items():
        cfg = baseline_config.copy()
        cfg['fc_units'] = fc_units
        results[f'fc_{name}'] = run_experiment(cfg, f'fc_{name}', epochs, device)

    # 不同损失函数 (L1/L2 正则化)
    loss_configs = {
        'ce_l1': {'l1_lambda': 1e-5, 'l2_lambda': 0},
        'ce_l2': {'l1_lambda': 0, 'l2_lambda': 1e-4},
        'ce_l1_l2': {'l1_lambda': 1e-5, 'l2_lambda': 1e-4},
    }
    for name, reg in loss_configs.items():
        cfg = baseline_config.copy()
        cfg.update(reg)
        results[f'loss_{name}'] = run_experiment(cfg, f'loss_{name}', epochs, device)

    # 不同激活函数
    for act in ['leaky_relu', 'gelu']:
        cfg = baseline_config.copy()
        cfg['activation'] = act
        results[f'act_{act}'] = run_experiment(cfg, f'act_{act}', epochs, device)

    # 不同优化器
    for opt_name in ['adam', 'adamw']:
        cfg = baseline_config.copy()
        cfg['optimizer'] = opt_name
        cfg['lr'] = 0.001
        results[f'opt_{opt_name}'] = run_experiment(cfg, f'opt_{opt_name}', epochs, device)

    # 无 BN/Dropout 对照
    cfg_no_extra = baseline_config.copy()
    cfg_no_extra['use_bn'] = False
    cfg_no_extra['use_dropout'] = False
    results['no_bn_dropout'] = run_experiment(cfg_no_extra, 'no_bn_dropout', epochs, device)

    # ResNet
    cfg_resnet = baseline_config.copy()
    cfg_resnet['model_cls'] = CIFAR10_ResNet
    results['resnet'] = run_experiment(cfg_resnet, 'resnet', epochs, device)

    # DeepCNN — baseline (BN + Dropout + Residual + AdamW)
    cfg_deep = {
        'model_cls': CIFAR10_DeepCNN,
        'stage_channels': (64, 128, 256, 512),
        'stage_depths': (2, 2, 3, 1),
        'fc_units': (512,),
        'activation': 'relu',
        'use_bn': True,
        'use_dropout': True,
        'dropout_p': 0.1,
        'optimizer': 'adamw',
        'lr': 0.001,
        'weight_decay': 5e-4,
        'augmentation': True,
        'patience': 15,
    }
    results['deepcnn_baseline'] = run_experiment(cfg_deep, 'deepcnn_baseline', epochs, device)

    # DeepCNN + MixUp
    cfg_deep_mixup = cfg_deep.copy()
    cfg_deep_mixup['mixup_alpha'] = 0.2
    results['deepcnn_mixup'] = run_experiment(cfg_deep_mixup, 'deepcnn_mixup', epochs, device)

    # DeepCNN + Label Smoothing + MixUp
    cfg_deep_ls_mix = cfg_deep.copy()
    cfg_deep_ls_mix['mixup_alpha'] = 0.2
    cfg_deep_ls_mix['label_smoothing'] = 0.1
    results['deepcnn_ls_mixup'] = run_experiment(cfg_deep_ls_mix, 'deepcnn_ls_mixup', epochs, device)

    # DeepCNN — 加深变体 (5 stages, 14 residual blocks)
    cfg_deep_large = cfg_deep.copy()
    cfg_deep_large['stage_channels'] = (64, 128, 256, 512, 512)
    cfg_deep_large['stage_depths'] = (2, 3, 4, 3, 2)
    results['deepcnn_large'] = run_experiment(cfg_deep_large, 'deepcnn_large', epochs, device)

    # 汇总
    print("\n" + "="*80)
    print("所有实验汇总")
    print("="*80)
    summary = {}
    for name, r in results.items():
        print(f"{name:25s} | Test Acc: {r['test_acc']:.4f} ({r['test_acc']*100:.2f}%) "
              f"| Params: {r['n_params']:,} | Best Val: {r['best_val_acc']:.4f}")
        summary[name] = {
            'test_acc': r['test_acc'],
            'best_val_acc': r['best_val_acc'],
            'n_params': r['n_params'],
        }

    summary_path = os.path.join(RESULTS_DIR, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n汇总保存至 {summary_path}")

    return results


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    if args.run_all:
        run_all_experiments(device, quick=args.quick)
    else:
        exp_name = args.name if args.name else generate_exp_name(args)
        config = build_config(args)
        epochs = 10 if args.quick else args.epochs

        print(f"实验名: {exp_name}")

        run_experiment(config, exp_name, epochs=epochs, device=device,
                       swanlab_project=args.swanlab_project)
