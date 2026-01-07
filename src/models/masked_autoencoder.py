import torch
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.decoder import Decoder
from src.models.encoder import Encoder

class MAE(nn.Module):
    def __init__(
            self,
            *,
            latent_dim=256,
            masking_ratio=0.75,
            img_size=32
            ):
        super().__init__()
        assert 0 < masking_ratio < 1, 'masking ratio must be between 0 and 1'

        self.masking_ratio = masking_ratio
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.latent_dim = latent_dim

    def random_masking(self, x):
        B, C, H, W = x.shape
        mask = torch.rand(B, 1, H, W, device=x.device) > self.masking_ratio
        mask = mask.float()
        x_masked = x * mask

        return x_masked, mask


    def forward(self, x):
        x_masked, mask = self.random_masking(x)
        latent = self.encoder(x_masked)
        reconstructed = self.decoder(latent)
        return reconstructed, x_masked, mask

    def compute_loss(self, original, reconstructed, mask):
        masked_region = 1 - mask
        loss = F.mse_loss(
            reconstructed * masked_region,
            original * masked_region,
            reduction='sum'
        )

        n_masked = masked_region.sum() # ile pikseli zamaskowanych
        loss = loss / (n_masked + 1e-8) # srednia tylko z zmaskowanych

        return loss

