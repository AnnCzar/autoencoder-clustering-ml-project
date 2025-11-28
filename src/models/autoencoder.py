import torch
import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(Autoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, x):
        z = self.encoder(x)  # kodowanie
        x_reconstructed = self.decoder(z)  # dekodowanie
        return x_reconstructed
