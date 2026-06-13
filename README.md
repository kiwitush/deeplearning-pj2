# DeepLearning PJ2
本项目是神经网络与深度学习课程 Project 2 的代码实现，完成以下两个任务：

1. **Task 1**: 在 CIFAR-10 上训练 CNN 分类网络
2. **Task 2**: 基于 VGG-A 分析 Batch Normalization 对训练过程及 Loss Landscape 的影响

---

## 项目结构

```
nn-pj2/
├── README.md
├── report.tex                             # LaTeX 实验报告
├── bib.bib                                # 参考文献
├── codes/
│   ├── cifar10/                           # Task 1: CIFAR-10 分类网络
│   │   ├── models.py                      # 模型定义 (CIFAR10_CNN / DeepCNN / ResNet)
│   │   ├── train.py                       # 训练循环、数据加载、工具函数
│   │   ├── experiments.py                 # 实验运行器 (CLI + 批量实验)
│   │   ├── ensemble_eval.py               # 模型集成评估
│   │   ├── visualization.py               # 可视化工具 (训练曲线 / 混淆矩阵 / 卷积核)
│   │   └── tsne_visualization.py          # t-SNE 特征可视化
│   └── VGG_BatchNorm/                     # Task 2: Batch Normalization
│       ├── models/
│       │   └── vgg.py                     # VGG_A / VGG_A_BatchNorm
│       ├── utils/
│       │   └── nn.py                      # 权重初始化
│       ├── data/
│       │   └── loaders.py                 # CIFAR-10 数据加载器
│       ├── VGG_Loss_Landscape.py          # 训练 + Loss Landscape 分析
│       └── test_vgg.py                    # 测试脚本：评估已保存模型
├── best_models/
│   ├── task1/                             # Task 1 模型权重 (.pth)
│   └── task2/                             # Task 2 模型权重 (.pth)
├── results/
│   ├── task1/                             
│   └── task2/                             
└── dataset/                               # CIFAR-10 数据集 (自动下载)
```

## 环境依赖

- **Python** >= 3.10
- **PyTorch** >= 2.0
- **NumPy** — 数值计算
- **Matplotlib** — 可视化
- **scikit-learn** — t-SNE 降维
- **tqdm** — 进度条

安装依赖：

```bash
pip install torch torchvision numpy matplotlib scikit-learn tqdm
```

---

## Task 1: CIFAR-10 分类网络

### 快速开始

所有脚本需在 `codes/cifar10` 目录下运行：

```bash
cd codes/cifar10

# Baseline (CNN, 全部默认值)
python experiments.py

# DeepCNN 基线
python experiments.py --model deepcnn --optimizer adamw --lr 0.001 --epochs 100

# 修改参数
python experiments.py --name my_exp --optimizer adam --lr 0.001
python experiments.py --activation gelu --epochs 100
python experiments.py --model deepcnn --mixup-alpha 0.2 --label-smoothing 0.1

# 批量运行所有预定义实验
python experiments.py --run-all
```

### 命令行参数

`experiments.py` 支持以下命令行参数，所有参数均为可选（有默认值）：

| 参数 | 类型 | 默认值 | 可选值 | 说明 |
|------|------|--------|--------|------|
| `--model` | str | `cnn` | `cnn`, `deepcnn`, `resnet` | 模型架构 |
| `--conv-channels` | int list | `32 64 128 256 512 512 512` | — | 每层 Conv 输出通道 |
| `--fc-units` | int list | `512 256` | — | FC 隐藏层单元数 |
| `--activation` | str | `relu` | `relu`, `leaky_relu`, `gelu` | 激活函数 |
| `--no-bn` | flag | False | — | 关闭 Batch Normalization |
| `--no-dropout` | flag | False | — | 关闭 Dropout |
| `--dropout-rate` | float | `0.1` | — | Dropout 概率 |
| `--no-gap` | flag | False | — | 关闭全局平均池化（使用 flatten） |
| `--stage-channels` | int list | `64 128 256 512` | — | DeepCNN Stage 通道数 |
| `--stage-depths` | int list | `2 2 3 1` | — | DeepCNN 每 Stage 残差块数 |
| `--num-blocks` | int list | `3 3 3` | — | ResNet 每 Stage block 数 |
| `--base-channels` | int | `16` | — | ResNet 基础通道数 |
| `--optimizer` | str | `sgd` | `sgd`, `adam`, `adamw` | 优化器 |
| `--lr` | float | `0.01` | — | 学习率 |
| `--weight-decay` | float | `5e-4` | — | 权重衰减 |
| `--momentum` | float | `0.9` | — | SGD 动量 |
| `--epochs` | int | `100` | — | 最大训练 epoch 数 |
| `--batch-size` | int | `128` | — | 批次大小 |
| `--patience` | int | `15` | — | 早停耐心值 |
| `--criterion` | str | `ce` | `ce`, `focal` | 损失函数 |
| `--focal-gamma` | float | `2.0` | — | Focal Loss γ 参数 |
| `--l1-lambda` | float | `0.0` | — | L1 正则化强度 |
| `--l2-lambda` | float | `0.0` | — | L2 正则化强度 |
| `--label-smoothing` | float | `0.0` | — | 标签平滑因子 |
| `--no-augment` | flag | False | — | 关闭数据增强 |
| `--mixup-alpha` | float | `0.0` | 建议 `0.2`~`0.4` | MixUp α（0=关闭） |
| `--cutmix-alpha` | float | `0.0` | 建议 `0.2` | CutMix α（0=关闭） |
| `--cutmix-prob` | float | `0.0` | 建议 `0.3` | CutMix 选择概率 |
| `--warmup-epochs` | int | `0` | 建议 `5` | LR Warmup epoch 数 |
| `--seed` | int | `42` | — | 随机种子 |
| `--name` | str | auto | — | 实验名称（影响保存文件名） |
| `--run-all` | flag | — | — | 批量运行所有预定义实验 |
| `--quick` | flag | — | — | 快速模式（10 epochs） |

### 实验矩阵

所有消融实验基于 DeepCNN 基线（seed=42, AdamW + MixUp + CutMix + Label Smoothing + Warmup）：

| 类别 | 变量 | 选项 |
|------|------|------|
| 优化器 | optimizer | SGD, Adam, AdamW |
| 激活函数 | activation | ReLU, LeakyReLU, GELU |
| FC 层大小 | fc_units | (256,128), (512,256), (1024,512,256) |
| 损失函数 | criterion | CrossEntropy, Focal Loss |
| 正则化 | L1/L2 | 无额外, L1(λ=10⁻⁵), L2(λ=10⁻⁴) |
| 模型集成 | seed | 42, 123, 456 (avg logits) |

### 数据预处理及增强
- 归一化：mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)
- 基础：RandomCrop(4) + RandomHorizontalFlip
- 高级（DeepCNN）：MixUp(α=0.2) + CutMix(α=0.2, prob=30%)

### 可视化

所有可视化均基于 DeepCNN Baseline (ensemble_s42.pth, seed=42)：

```bash
cd codes/cifar10

# 批量训练曲线
python visualization.py --curves

# 混淆矩阵
python visualization.py --confusion

# 卷积核可视化
python visualization.py --conv

# 可组合使用
python visualization.py --confusion --conv

# t-SNE 特征可视化
python tsne_visualization.py
```

生成图片：
- `training_curves_{exp}.png` — 各实验的训练/验证曲线
- `confusion_matrix.png` — DeepCNN Baseline 混淆矩阵
- `conv_filters.png` — 第一层卷积核可视化
- `tsne.png` — 256 维 FC 特征的 t-SNE 降维

---

## Task 2: Batch Normalization

### VGG-A 结构 (适配 CIFAR-10)

8 个卷积层 + 3 个全连接层，共 11 层可学习参数。分类头基于 CIFAR-10 适配为 512→512→10。

VGG_A_BatchNorm 在每个 Conv 和 ReLU 之间插入 BatchNorm2d 层。

### Loss Landscape 分析

使用四个固定学习率 `[2×10⁻³, 10⁻³, 5×10⁻⁴, 10⁻⁴]`，分别训练 VGG_A 和 VGG_A_BatchNorm，记录每步 loss。对每个 step 计算所有 LR 下的 max/min loss，用 `plt.fill_between` 填充范围，绘制 gap 曲线 (max−min) 比较两种模型的 loss landscape 平滑度。

### 快速开始

所有脚本需在 `codes/VGG_BatchNorm/` 目录下运行：

```bash
cd codes/VGG_BatchNorm

# 完整实验 (VGG-A vs VGG-A+BN 对比 + Loss Landscape)
python VGG_Loss_Landscape.py

# 仅对比训练
python VGG_Loss_Landscape.py --comparison

# 仅 Loss Landscape 分析
python VGG_Loss_Landscape.py --landscape

# 自定义 epochs
python VGG_Loss_Landscape.py --epochs 30 --landscape-epochs 30
```

### 测试已保存模型

```bash
python test_vgg.py --model VGG_A
python test_vgg.py --model VGG_A_BatchNorm
```

---

## 模型权重下载地址
- 模型权重：[百度网盘](https://pan.baidu.com/s/16Ww_UgJajisZF4w9FrzlcA)（提取码：6h7g）
