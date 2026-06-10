# CIFAR-10 图像分类与 Batch Normalization 优化分析

本项目是"神经网络与深度学习"课程 Project 2 的代码实现，使用 PyTorch 完成以下两个任务：

1. **Task 1**: 在 CIFAR-10 上训练 CNN 分类网络，探索不同架构、优化器和正则化策略
2. **Task 2**: 基于 VGG-A 分析 Batch Normalization 对训练过程及 Loss Landscape 的影响

---

## 项目结构

```
nn-pj2/
├── README.md
├── report.tex                        # LaTeX 实验报告（编译后提交 PDF）
├── codes/
│   ├── cifar10/                # Task 1: CIFAR-10 分类网络
│   │   ├── models.py                 # CNN 模型定义 (CIFAR10_CNN / ResNet)
│   │   ├── train.py                  # 训练循环、数据加载、工具函数
│   │   ├── experiments.py            # 完整实验脚本（基线 + 消融实验）
│   │   └── visualization.py          # 滤波器可视化、训练曲线、混淆矩阵
│   └── VGG_BatchNorm/                # Task 2: Batch Normalization
│       ├── models/
│       │   └── vgg.py                # VGG_A / VGG_A_BatchNorm / VGG_A_Dropout
│       ├── utils/
│       │   └── nn.py                 # 权重初始化
│       ├── data/
│       │   └── loaders.py            # CIFAR-10 数据加载器
│       ├── VGG_Loss_Landscape.py     # 训练 + Loss Landscape 分析（完整实验）
│       └── test_vgg.py               # 测试脚本：评估已保存模型
├── best_models/
│   ├── task1/                          # Task 1 模型权重
│   └── task2/                          # Task 2 模型权重
├── results/
│   ├── task1/                          # Task 1 图片（训练曲线、卷积核等）
│   └── task2/                          # Task 2 图片（对比曲线、Loss Landscape 等）
└── dataset/                          # CIFAR-10 数据集（自动下载）
```

## 环境依赖

- **Python** >= 3.10
- **PyTorch** >= 2.0
- **NumPy** — 数值计算
- **Matplotlib** — 可视化
- **tqdm** — 进度条

安装依赖：

```bash
pip install torch torchvision numpy matplotlib tqdm
```

## Task 1: CIFAR-10 分类网络

### 快速开始

```bash
cd codes/cifar10

# Baseline（全部默认值）
python experiments.py

# 快速测试（10 epochs）
python experiments.py --quick

# 灵活修改参数
python experiments.py --name my_exp --optimizer adam --lr 0.001
python experiments.py --activation gelu --epochs 80
python experiments.py --no-bn --no-dropout --name ablation
python experiments.py --model resnet --num-blocks 5 5 5

# 批量运行所有预定义实验
python experiments.py --run-all
```

### CLI 参数一览

| 类别 | 参数 | 默认值 | 说明 |
|------|------|--------|------|
| Model | `--model` | cnn | cnn / resnet |
| | `--conv-channels` | 32 64 128 256 512 512 512 | 卷积通道数 |
| | `--fc-units` | 512 256 | FC 隐藏单元 |
| | `--activation` | relu | relu / leaky_relu / gelu |
| | `--no-bn` | False | 关闭 BatchNorm |
| | `--no-dropout` | False | 关闭 Dropout |
| | `--no-gap` | False | 关闭 GAP（用 flatten） |
| | `--num-blocks` | 3 3 3 | ResNet 每 stage 的 block 数 |
| | `--base-channels` | 16 | ResNet 基础通道数 |
| Optimizer | `--optimizer` | sgd | sgd / adam / adamw |
| | `--lr` | 0.01 | 学习率 |
| | `--weight-decay` | 5e-4 | 权重衰减 |
| | `--momentum` | 0.9 | SGD 动量 |
| Training | `--epochs` | 50 | 最大训练轮数 |
| | `--batch-size` | 128 | 批次大小 |
| | `--patience` | 15 | 早停耐心值 |
| | `--l1-lambda` | 0.0 | L1 正则化强度 |
| | `--l2-lambda` | 0.0 | L2 正则化强度 |
| | `--no-augment` | False | 关闭数据增强 |
| Other | `--name` | auto | 实验名称（不指定则自动生成） |
| | `--seed` | 42 | 随机种子 |
| | `--quick` | False | 10 epoch 快速测试 |
| | `--run-all` | False | 批量预定义实验 |

### 模型结构

**CIFAR10_CNN (Baseline)** — Conv→BN→ReLU ×7，每 2 层 Pool，最后 GAP：

```
  Conv(3→32) → BN → ReLU
  Conv(32→64) → BN → ReLU → MaxPool(32→16)
  Conv(64→128) → BN → ReLU
  Conv(128→256) → BN → ReLU → MaxPool(16→8)
  Conv(256→512) → BN → ReLU
  Conv(512→512) → BN → ReLU → MaxPool(8→4)
  Conv(512→512) → BN → ReLU
  GAP(4×4→1×1) → FC(512) → ReLU → Dropout(0.3) → FC(256) → ReLU → Output(10)
```

设计要点：
- **克制降采样**：仅 3 次 MaxPool（32→16→8→4），最后保留 4×4 空间信息
- **GAP 收尾**：用 Global Average Pooling 替代大 FC 展开，大幅减少参数量
- **Kaiming 初始化**：所有 Conv / Linear 层用 `kaiming_normal_`（适配 ReLU）

**CIFAR10_ResNet** — 标准 CIFAR ResNet：

```
  Stem: Conv(3→16) → BN → ReLU
  Stage1: N × BasicBlock(16→16)         [32×32]
  Stage2: N × BasicBlock(16→32, stride=2) [16×16]
  Stage3: N × BasicBlock(32→64, stride=2) [8×8]
  GAP → Linear(64→10)
```

### 实验矩阵

| 实验类别 | 变量 | 选项 |
|----------|------|------|
| 滤波器数量 | conv_channels | narrow (16,32,64,128,256,256,256), baseline (32,64,128,256,512,512,512), wide (64,128,256,512,512,512,512) |
| FC 层大小 | fc_units | small (256,128), baseline (512,256), large (1024,512,256) |
| 损失函数 | 正则化 | CE only, CE+L1, CE+L2, CE+L1+L2 |
| 激活函数 | activation | ReLU, LeakyReLU, GELU |
| 优化器 | optimizer | SGD (momentum=0.9, nesterov), Adam, AdamW |
| 正则化组件 | BN + Dropout | With BN+Dropout vs Without |
| 网络架构 | model | CNN (baseline) vs ResNet |

### 数据增强

- 随机裁剪（32×32，padding=4）
- 随机水平翻转
- 归一化：mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)

### 可视化

```bash
python visualization.py
```

生成图片：
- `training_curves_{exp}.png` — 各实验的训练/验证曲线
- `comparison_test_acc.png` — 所有实验的测试准确率对比
- `conv_filters.png` — 第一层卷积核可视化
- `confusion_matrix.png` — 混淆矩阵

---

## Task 2: Batch Normalization

### 快速开始

所有脚本需在 `codes/VGG_BatchNorm/` 目录下运行：

```bash
cd codes/VGG_BatchNorm
```

### 运行完整实验

```bash
# 完整实验（VGG-A vs VGG-A+BN 对比 + Loss Landscape）
python VGG_Loss_Landscape.py

# 只做 VGG-A vs VGG-A+BN 对比
python VGG_Loss_Landscape.py --comparison

# 只做 Loss Landscape 分析
python VGG_Loss_Landscape.py --landscape

# 自定义 epochs
python VGG_Loss_Landscape.py --epochs 30 --landscape-epochs 30
```

### VGG-A 结构 (适配 CIFAR-10: 32×32×3)

| Stage | VGG-A (no BN) | VGG_A_BatchNorm |
|-------|---------------|-----------------|
| 1 | Conv(3,64) → ReLU → Pool | Conv(3,64) → **BN** → ReLU → Pool |
| 2 | Conv(64,128) → ReLU → Pool | Conv(64,128) → **BN** → ReLU → Pool |
| 3 | Conv(128,256) → ReLU → Conv(256,256) → ReLU → Pool | Conv(128,256) → **BN** → ReLU → Conv(256,256) → **BN** → ReLU → Pool |
| 4 | Conv(256,512) → ReLU → Conv(512,512) → ReLU → Pool | Conv(256,512) → **BN** → ReLU → Conv(512,512) → **BN** → ReLU → Pool |
| 5 | Conv(512,512) → ReLU → Conv(512,512) → ReLU → Pool | Conv(512,512) → **BN** → ReLU → Conv(512,512) → **BN** → ReLU → Pool |
| FC | Linear(512→512) → ReLU → Linear(512→512) → ReLU → Linear(512→10) | 同左 |

### Loss Landscape 分析

使用多个学习率 `[2e-3, 1e-3, 5e-4, 1e-4]`（跨度 20 倍），在 5000 张随机子集上分别用 SGD+momentum=0.9 训练两个模型，记录每步 loss。对每个 step 计算所有 LR 下 max/min loss，用 `plt.fill_between` 填充范围，同时绘制 gap 曲线（max-min）。BN 通过约束各层激活统计量、防止层间尺度失衡，使 loss landscape 更平滑。

### 测试已保存模型

```bash
python test_vgg.py --model VGG_A
python test_vgg.py --model VGG_A_BatchNorm
```

---

## 实验结果

详细实验数据与报告框架见 [report.tex](report.tex)，编译后生成 PDF 提交。

## 作者信息

- 姓名：郑乐怡
- 学号：23307140100
- GitHub：[kiwitush/nn-pj2](https://github.com/kiwitush/nn-pj2)
- 模型权重：[Google Drive](https://drive.google.com/drive/folders/...)（上传中）
"# deeplearning-pj2" 
