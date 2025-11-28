import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, latent_dim=256):

        super(Decoder, self).__init__()
        self.latent_dim = latent_dim

        # latent vector -> tensor 128x8x8
        self.fc = nn.Linear(latent_dim, 128*8*8)

        # blok 1: 8x8 -> 16x16
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        # blok 2: 16x16 -> 32x32
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        #  32x32 -> 32x32x3
        self.conv3 = nn.Conv2d(32, 3, kernel_size=3, padding=1)

    def forward(self, x):
        # latent vector -> tensor 128x8x8
        x = self.fc(x)
        x = x.view(-1, 128, 8, 8)

        # blok 1
        x = self.up1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        # blok 2
        x = self.up2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = torch.sigmoid(x)  # to ensure output is in [0, 1]

        return x