import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', '..', 'data', 'raw')
os.makedirs(DATA_DIR, exist_ok=True)

trainset = torchvision.datasets.CIFAR10(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
)

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=8,
    shuffle=True
)

images, labels = next(iter(trainloader))

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')

plt.figure(figsize=(12, 6))
imshow(torchvision.utils.make_grid(images))
plt.title(' | '.join(trainset.classes[label] for label in labels))
plt.tight_layout()
plt.show()


print(f"Classes: {trainset.classes}")
print(f"Dataset size: {len(trainset)}")