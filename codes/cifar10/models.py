"""
Task 1: CIFAR-10 分类模型

模型:
  - CIFAR10_CNN: 灵活可配的 Conv→BN→ReLU→Pool→GAP/Flatten→FC
  - CIFAR10_CNN_Small: 轻量版，快速实验用
  - CIFAR10_DeepCNN: 加深版，Stage 结构 + 残差连接
  - CIFAR10_ResNet: 标准 CIFAR ResNet (BasicBlock + GAP)

CIFAR-10 (32×32) 设计要点:
  - Stem: 3×3, stride=1, 无初始 MaxPool → 保持 32×32 分辨率
  - Backbone: 4 次下采样 (32→16→8→4→2 或 32→16→8→4), GAP 避免巨量 FC
  - Init: Kaiming normal for all ReLU-activated layers
"""

import torch
import torch.nn as nn


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class CIFAR10_CNN(nn.Module):
    """灵活 CNN，用于 CIFAR-10。

    结构: N×(Conv3×3→BN?→Act)→每 2 层 Pool→GAP?→FC→10

    Args:
        conv_channels: 每层 conv 的输出通道, 如 [32,64,128,256,512,512,512]
        fc_units: FC 隐藏单元, 如 [512, 256]
        activation: 'relu' | 'leaky_relu' | 'gelu'
        use_bn: 是否在 conv 后加 BN
        use_dropout: 是否在 FC 后加 Dropout (BN 后不加，避免方差偏移)
        dropout_p: Dropout 概率
        use_gap: 用 GAP 替代 Flatten→大 FC
        num_classes: 输出类别数
    """

    def __init__(self, conv_channels=(32, 64, 128, 256, 512, 512, 512),
                 fc_units=(512, 256),
                 activation='relu',
                 use_bn=True,
                 use_dropout=True,
                 dropout_p=0.1,
                 use_gap=True,
                 num_classes=10):
        super().__init__()
        self.use_gap = use_gap

        in_ch = 3
        self.conv_blocks = nn.ModuleList()
        for out_ch in conv_channels:
            block = nn.Sequential()
            block.add_module('conv', nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
            if use_bn:
                block.add_module('bn', nn.BatchNorm2d(out_ch))
            block.add_module('act', self._get_activation(activation))
            self.conv_blocks.append(block)
            in_ch = out_ch

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 计算 FC 输入维度 (GAP 时 = 末层通道数, 否则 = flatten 大小)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 32, 32)
            dummy = self._forward_conv(dummy)
            if use_gap:
                self._flatten_size = out_ch
            else:
                self._flatten_size = dummy.view(1, -1).size(1)

        if use_gap:
            self.gap = nn.AdaptiveAvgPool2d((1, 1))

        fc_layers = []
        prev = self._flatten_size
        for units in fc_units:
            fc_layers.append(nn.Linear(prev, units))
            if use_dropout:
                fc_layers.append(nn.Dropout(dropout_p))
            fc_layers.append(self._get_activation(activation))
            prev = units
        fc_layers.append(nn.Linear(prev, num_classes))
        self.classifier = nn.Sequential(*fc_layers)

        self._init_weights()

    def _get_activation(self, name):
        name = name.lower()
        if name == 'relu':
            return nn.ReLU(inplace=True)
        elif name == 'leaky_relu':
            return nn.LeakyReLU(0.1, inplace=True)
        elif name == 'gelu':
            return nn.GELU()
        else:
            raise ValueError(f"未知激活函数: {name}")

    def _forward_conv(self, x):
        n = len(self.conv_blocks)
        for i, block in enumerate(self.conv_blocks):
            x = block(x)
            # 每 2 层 pool 一次，最后一层不 pool (保留 4×4 特征图)
            if (i + 1) % 2 == 0 and i != n - 1:
                x = self.pool(x)
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        if self.use_gap:
            x = self.gap(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class CIFAR10_CNN_Small(nn.Module):
    """轻量 CNN，快速实验用。

    结构: 3 stage, 每 stage 2 层 conv + pooling
    """

    def __init__(self, activation='relu', use_bn=True, num_classes=10):
        super().__init__()

        def conv_block(in_ch, out_ch):
            layers = [nn.Conv2d(in_ch, out_ch, 3, padding=1)]
            if use_bn:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True) if activation == 'relu' else
                          nn.LeakyReLU(0.1, inplace=True) if activation == 'leaky_relu' else
                          nn.GELU())
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            conv_block(3, 32),
            conv_block(32, 32),
            nn.MaxPool2d(2, 2),

            conv_block(32, 64),
            conv_block(64, 64),
            nn.MaxPool2d(2, 2),

            conv_block(64, 128),
            conv_block(128, 128),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

        self._init_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class ResidualBlock(nn.Module):
    """残差块: Conv→BN→ReLU→Conv→BN→+shortcut→ReLU。

    stride>1 或通道数变化时，用 1×1 Conv 对齐 shortcut 维度。
    """

    def __init__(self, in_ch, out_ch, stride=1, use_bn=True, activation='relu'):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch) if use_bn else nn.Identity()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch) if use_bn else nn.Identity()

        if in_ch != out_ch or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch) if use_bn else nn.Identity(),
            )
        else:
            self.shortcut = nn.Identity()

        if activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.act = nn.LeakyReLU(0.1, inplace=True)
        elif activation == 'gelu':
            self.act = nn.GELU()
        else:
            raise ValueError(f"未知激活函数: {activation}")

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + self.shortcut(x))


class CIFAR10_DeepCNN(nn.Module):
    """加深 CNN + 残差连接 + BN + Dropout (仅 FC)。

    结构:
      Stage×K: N 个 ResidualBlock (Conv→BN→ReLU→Conv→BN→+shortcut→ReLU)
               Stage 间 stride=2 下采样 (32→16→8→4)
      Head:    GAP → FC → Dropout → ReLU → FC → 10

    BN 覆盖所有 conv 层, Dropout 仅用于 FC (避免 BN-Dropout 方差偏移)。
    残差连接使 20+ 层网络可稳定训练。

    Args:
        stage_channels: 每 stage 输出通道, 如 [64, 128, 256, 512]
        stage_depths:   每 stage 残差块数量, 如 [2, 3, 4, 2]
        fc_units:       FC 隐藏单元, 如 [512]
        activation:     'relu' | 'leaky_relu' | 'gelu'
        use_bn:         是否在 conv 后加 BN
        use_dropout:    是否在 FC 后加 Dropout
        dropout_p:      Dropout 概率
        num_classes:    输出类别数
    """

    def __init__(self, stage_channels=(64, 128, 256, 512),
                 stage_depths=(2, 2, 3, 1),
                 fc_units=(512,),
                 activation='relu',
                 use_bn=True,
                 use_dropout=True,
                 dropout_p=0.1,
                 num_classes=10):
        super().__init__()

        in_ch = 3
        self.stages = nn.ModuleList()

        for stage_idx, (out_ch, depth) in enumerate(zip(stage_channels, stage_depths)):
            blocks = []
            for blk_idx in range(depth):
                # 除 Stage 0 外, 每 stage 的首个 block 用 stride=2 下采样
                stride = 2 if (blk_idx == 0 and stage_idx > 0) else 1
                blocks.append(ResidualBlock(in_ch, out_ch, stride, use_bn, activation))
                in_ch = out_ch
            self.stages.append(nn.Sequential(*blocks))

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        fc_layers = []
        prev = stage_channels[-1]
        for units in fc_units:
            fc_layers.append(nn.Linear(prev, units))
            if use_dropout:
                fc_layers.append(nn.Dropout(dropout_p))
            fc_layers.append(self._get_activation(activation))
            prev = units
        fc_layers.append(nn.Linear(prev, num_classes))
        self.classifier = nn.Sequential(*fc_layers)

        self._init_weights()

    def _get_activation(self, name):
        name = name.lower()
        if name == 'relu':
            return nn.ReLU(inplace=True)
        elif name == 'leaky_relu':
            return nn.LeakyReLU(0.1, inplace=True)
        elif name == 'gelu':
            return nn.GELU()
        else:
            raise ValueError(f"未知激活函数: {name}")

    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class BasicBlock(nn.Module):
    """标准 ResNet BasicBlock (两个 3×3 Conv)。

    stride>1 或通道数变化时，用 1×1 Conv shortcut 对齐维度。
    """

    def __init__(self, in_ch, out_ch, stride=1, use_bn=True, activation='relu'):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch) if use_bn else nn.Identity()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch) if use_bn else nn.Identity()

        self.use_shortcut = (in_ch != out_ch) or (stride != 1)
        if self.use_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch) if use_bn else nn.Identity(),
            )
        else:
            self.shortcut = nn.Identity()

        self.act = nn.ReLU(inplace=True) if activation == 'relu' else \
                    nn.LeakyReLU(0.1, inplace=True) if activation == 'leaky_relu' else \
                    nn.GELU()

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.act(out + self.shortcut(x))
        return out


class CIFAR10_ResNet(nn.Module):
    """标准 ResNet for CIFAR-10。

    结构:
      Stem:   Conv3×3 → BN → ReLU (stride=1, 无 pooling → 32×32)
      Stage1: N₁ BasicBlock, width=w   (32×32)
      Stage2: N₂ BasicBlock, width=2w  (16×16, 首个 block stride=2)
      Stage3: N₃ BasicBlock, width=4w  (8×8,  首个 block stride=2)
      Head:   GAP → FC → 10

    默认 ResNet-20: num_blocks=(3,3,3), base_channels=16。
    """

    def __init__(self, num_blocks=(3, 3, 3), base_channels=16,
                 activation='relu', use_bn=True, num_classes=10):
        super().__init__()
        self.in_channels = base_channels

        # Stem: 3×3 conv, no pooling
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True) if activation == 'relu' else
            nn.LeakyReLU(0.1, inplace=True) if activation == 'leaky_relu' else
            nn.GELU(),
        )

        # 3 stages, 每次通道翻倍、空间减半
        self.stage1 = self._make_stage(base_channels, num_blocks[0], stride=1,
                                       use_bn=use_bn, activation=activation)
        self.stage2 = self._make_stage(base_channels * 2, num_blocks[1], stride=2,
                                       use_bn=use_bn, activation=activation)
        self.stage3 = self._make_stage(base_channels * 4, num_blocks[2], stride=2,
                                       use_bn=use_bn, activation=activation)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_channels * 4, num_classes)

        self._init_weights()

    def _make_stage(self, out_ch, num_blocks, stride, use_bn, activation):
        layers = [BasicBlock(self.in_channels, out_ch, stride, use_bn, activation)]
        self.in_channels = out_ch
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_ch, out_ch, 1, use_bn, activation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


if __name__ == '__main__':
    for model_cls in [CIFAR10_CNN, CIFAR10_CNN_Small, CIFAR10_DeepCNN, CIFAR10_ResNet]:
        model = model_cls()
        n_params = count_parameters(model)
        print(f"{model_cls.__name__:25s} 参数量: {n_params:,}")

    # 不同 CNN 配置
    configs = [
        {'conv_channels': (32, 64, 128), 'fc_units': (256, 128), 'name': '窄网络'},
        {'conv_channels': (64, 128, 256, 512), 'fc_units': (512, 256), 'name': '宽网络'},
        {'conv_channels': (32, 64, 128, 256), 'fc_units': (512, 256, 128), 'name': '深FC'},
    ]
    for cfg in configs:
        name = cfg.pop('name')
        model = CIFAR10_CNN(**cfg)
        print(f"CIFAR10_CNN ({name:6s}) 参数量: {count_parameters(model):,}")

    # ResNet 变体
    for blocks in [(2, 2, 2), (3, 3, 3), (5, 5, 5)]:
        model = CIFAR10_ResNet(num_blocks=blocks)
        print(f"CIFAR10_ResNet blocks={blocks} 参数量: {count_parameters(model):,}")
