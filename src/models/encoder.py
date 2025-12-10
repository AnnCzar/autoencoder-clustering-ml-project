import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, latent_dim=256):

        super(Encoder, self).__init__()

        # conv11
        self.conv11 = nn.Conv2d(1, 32, kernel_size = 3, padding=1)
        self.bn11 = nn.BatchNorm2d(32)

        # conv12
        self.conv12 = nn.Conv2d(32,32, 3, padding=1)
        self.bn12 = nn.BatchNorm2d(32)
        self.mp12 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        #conv21
        self.conv21 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn21 = nn.BatchNorm2d(64)
        # conv22
        self.conv22 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn22 = nn.BatchNorm2d(64)
        self.mp22 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # conv31
        self.conv31 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn31 = nn.BatchNorm2d(128)
        # conv32
        self.conv32 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn32 = nn.BatchNorm2d(128)
        # self.ap32 = nn.AvgPool2d(kernel_size=7, stride=7, padding=0)       
        self.poo31 = nn.AdaptiveAvgPool2d((1, 1)) # change for datasets with different input size COVID-19
        
        # latent space
        self.fc = nn.Linear(128, latent_dim)
        self.latent_dim = latent_dim
    

    def forward(self, x):

        # conv11
        x = self.conv11(x)
        x = self.bn11(x)
        x = F.relu(x)

        # conv12
        x = self.conv12(x)
        x = self.bn12(x)
        x = F.relu(x)
        x = self.mp12(x)

        # conv21
        x = self.conv21(x)
        x = self.bn21(x)
        x = F.relu(x)

        # conv22
        x = self.conv22(x)
        x = self.bn22(x)
        x = F.relu(x)
        x = self.mp22(x)
    

        # conv31
        x = self.conv31(x)
        x = self.bn31(x)
        x = F.relu(x)
    
        # conv32
        x = self.conv32(x)
        x = self.bn32(x)
        x = F.relu(x)


        x = self.poo31(x)

        # flatten
        x = torch.flatten(x, 1)
        # latent space
        x = self.fc(x)
        return x

