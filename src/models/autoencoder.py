import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.decoder import Decoder
from src.models.encoder import Encoder


class Autoencoder(nn.Module):
    def __init__(self, latent_dim = 256):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.latent_dim = latent_dim
    
    def forward(self, x):
        """
        x: obraz [batch_size, 3, 32, 32]
        return: zrekonstruowany obraz [batch_size, 3, 32, 32]
        """
        z = self.encoder(x)  # kodowanie -- kompresja do  [batch_size, 256]
        x_reconstructed = self.decoder(z)  # dekodowanie -- odtworzenie do [batch_size, 3, 32, 32]
        return x_reconstructed

    def encode(self, x):
        """
        Pomocnicza funkcja do wyciągania cech
        x: obraz [batch_size, 3, 32, 32]
        return: wektor cech [batch_size, 256]
        """
        return self.encoder(x)