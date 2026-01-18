
import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelDecoder(nn.Module):

    def __init__(self, latent_dim=256):
        super().__init__()

        self.fc = nn.Linear(latent_dim, 128)
        

        self.up0 = nn.Upsample(scale_factor=8, mode='nearest')
        self.conv0 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn0 = nn.BatchNorm2d(128)
        
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        

        self.conv3 = nn.Conv2d(32, 3, kernel_size=3, padding=1)

        self.dropout = nn.Dropout(p=0.3)
    
    def forward(self, x):

        x = self.fc(x)
        x = x.view(-1, 128, 1, 1)  # (B, 128, 1, 1)
        
        x = self.up0(x)
        x = self.conv0(x)
        x = self.bn0(x)
        x = F.relu(x)  # (B, 128, 8, 8)
        
        x = self.up1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)  # (B, 64, 16, 16)
        
        x = self.up2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)  # (B, 32, 32, 32)
        
        x = self.conv3(x)  # (B, 3, 32, 32)
        # x = torch.sigmoid(x)  # [0, 1]  # zakomentować jak maskujemy patche
        
        return x


class FeatureDecoder(nn.Module):

    def __init__(self, latent_dim=256):
        super().__init__()
        
        self.fc = nn.Linear(latent_dim, 128)
        

        self.up0 = nn.Upsample(scale_factor=8, mode='nearest')
        self.conv0 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn0 = nn.BatchNorm2d(128)
        

        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        

        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        

        self.fc_out = nn.Linear(32, latent_dim)
    
    def forward(self, x):

        x = self.fc(x)
        x = x.view(-1, 128, 1, 1)  # (B, 128, 1, 1)
        

        x = self.up0(x)
        x = self.conv0(x)
        x = self.bn0(x)
        x = F.relu(x)  # (B, 128, 8, 8)
        
        x = self.up1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)  # (B, 64, 16, 16)
        

        x = self.up2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)  # (B, 32, 32, 32)
        

        x = F.adaptive_avg_pool2d(x, (1, 1))  # (B, 32, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 32)
        

        x = self.fc_out(x)  
        
        return x
