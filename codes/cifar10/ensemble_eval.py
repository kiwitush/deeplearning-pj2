"""模型集成评估: 加载 3 个模型, 平均 logits 后投票"""
import os, sys, torch, numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import CIFAR10_DeepCNN
from train import get_device

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(ROOT, 'best_models', 'task1')
SEEDS = [42, 123, 456]

device = get_device()

# 加载三个模型
models = []
for seed in SEEDS:
    path = os.path.join(MODEL_DIR, f'ensemble_s{seed}.pth')
    model = CIFAR10_DeepCNN(
        stage_channels=(64, 128, 256, 512), stage_depths=(2, 2, 3, 1),
        fc_units=(512, 256), dropout_p=0.1,
    )
    model.load_state_dict(torch.load(path, map_location=device))
    model = model.to(device).eval()
    models.append(model)
    print(f'加载: ensemble_s{seed}.pth')

# 测试集
normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                 std=[0.2470, 0.2435, 0.2616])
test_set = datasets.CIFAR10(
    root=os.path.join(ROOT, 'dataset'), train=False,
    transform=transforms.Compose([transforms.ToTensor(), normalize]))
test_loader = DataLoader(test_set, batch_size=128, shuffle=False,
                         num_workers=0)

# 集成评估
correct, total = 0, 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        logits = torch.stack([m(x) for m in models]).mean(dim=0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += x.size(0)

print(f'\n集成测试准确率: {correct/total:.4f} ({correct/total*100:.2f}%)')

# 各模型单独评估
for i, model in enumerate(models):
    correct_i, total_i = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            correct_i += (model(x).argmax(dim=1) == y).sum().item()
            total_i += x.size(0)
    print(f'  单模型 s{SEEDS[i]}: {correct_i/total_i:.4f} ({correct_i/total_i*100:.2f}%)')
