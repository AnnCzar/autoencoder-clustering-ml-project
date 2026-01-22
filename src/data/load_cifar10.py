import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import random as random_module
from torch.utils.data import Subset

def get_data_dir():
    """Zwraca ścieżkę do folderu z danymi"""
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, '..', '..', 'data', 'raw')
    os.makedirs(DATA_DIR, exist_ok=True)
    return DATA_DIR

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# DATA_DIR = os.path.join(BASE_DIR, '..', '..', 'data', 'raw')
# os.makedirs(DATA_DIR, exist_ok=True)

def get_cifar10_loaders(batch_size=64):
    """
    Zwraca train_loader, test_loader i validate_loader dla CIFAR-10

    Args:
        batch_size: Ile obrazów w jednym batchu

    Returns:
        train_loader, test_loader, validate_loader
    """
    DATA_DIR = get_data_dir()

    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_set_full = torchvision.datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=transform
    )

    train_size = int(0.8 * len(train_set_full))
    test_size = len(train_set_full) - train_size
    train_set, validate_set = torch.utils.data.random_split(train_set_full, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True
    )

    validate_loader = torch.utils.data.DataLoader(
        validate_set,
        batch_size=batch_size,
        shuffle=True
    )

    test_set = torchvision.datasets.CIFAR10(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, validate_loader, test_loader


def create_and_load_subset_c10(num_classes, batch_size=64, selected_classes=None, seed=None):
    total_classes = 10
    DATA_DIR = get_data_dir()

    if seed is not None:
        random_module.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    full_dataset = torchvision.datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=transform
    )

    full_test_ds = torchvision.datasets.CIFAR10(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=transform
    )

    all_labels = np.array(full_dataset.targets)
    all_labels_test = np.array(full_test_ds.targets)

    if selected_classes is None:
        selected_classes = random_module.sample(range(total_classes), num_classes)
        print(f"Wylosowano nowe klasy: {selected_classes}")
    else:
        print(f"UÅ¼ywam podanych klas: {selected_classes}")


    mask = np.isin(all_labels, selected_classes)
    mask_test = np.isin(all_labels_test, selected_classes)

    subset_indices = np.where(mask)[0]
    subset_indices_test = np.where(mask_test)[0]

    subset = Subset(full_dataset, subset_indices)
    subset_test = Subset(full_test_ds, subset_indices_test)

    train_size = int(0.8 * len(subset))
    test_size = len(subset) - train_size
    train_set, val_set = torch.utils.data.random_split(subset, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        subset_test,
        batch_size=batch_size,
        shuffle=False,
    )

    return subset, selected_classes, train_loader, val_loader, test_loader

# if __name__ == '__main__':
#     train_loader, validate_loader, test_loader = get_cifar10_loaders()
#
#     images, labels = next(iter(train_loader))
#
#     def imshow(img):
#         img = img / 2 + 0.5
#         npimg = img.numpy()
#         plt.imshow(np.transpose(npimg, (1, 2, 0)))
#         plt.axis('off')

    # plt.figure(figsize=(12, 6))
    # imshow(torchvision.utils.make_grid(images))
    # plt.title(' | '.join(trainset.classes[label] for label in labels))
    # plt.tight_layout()
    # plt.show()
    #
    #
    # print(f"Classes: {trainset.classes}")
    # print(f"Dataset size: {len(trainset)}")