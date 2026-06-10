"""
Data loaders
"""
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import torchvision.datasets as datasets


def get_cifar_loader(root='../data/', batch_size=128, train=True, shuffle=True, num_workers=4, n_items=-1):
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])

    data_transforms = transforms.Compose(
        [transforms.ToTensor(),
        normalize])

    dataset = datasets.CIFAR10(root=root, train=train, download=True, transform=data_transforms)
    if n_items > 0:
        # Shuffle indices before taking subset to avoid class bias
        rng = np.random.RandomState(2020)
        indices = rng.permutation(len(dataset))[:n_items]
        dataset = Subset(dataset, indices)

    # drop_last only for training to avoid BN instability on incomplete batches
    drop_last = train

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                        num_workers=num_workers, drop_last=drop_last)

    return loader

if __name__ == '__main__':
    train_loader = get_cifar_loader()
    for X, y in train_loader:
        print(X[0])
        print(y[0])
        print(X[0].shape)
        img = np.transpose(X[0], [1,2,0])
        plt.imshow(img*0.5 + 0.5)
        plt.savefig('sample.png')
        print(X[0].max())
        print(X[0].min())
        break