import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from src.models.cmae_decoders import PixelDecoder, FeatureDecoder
from src.models.encoder import Encoder


class ProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class CMAE(nn.Module):

    def __init__(
            self,
            latent_dim=256,
            masking_ratio=0.75,
            momentum=0.996, # parameter for updating the target encoder
            temperature=0.07, # temperature for contrastive loss
            contrastive_loss_weight=0.5, # weight for contrastive loss
    ):
        
        super().__init__()

        self.masking_ratio = masking_ratio
        self.momentum = momentum
        self.temperature = temperature
        self.contrastive_loss_weight = contrastive_loss_weight


        # online branch -- student-- with gradients
        self.online_encoder = Encoder(latent_dim)
        self.pixel_decoder = PixelDecoder(latent_dim)       # rekonstrukcja pikseli
        self.feature_decoder = FeatureDecoder(latent_dim)   # ekstrakcja cech
        self.online_projection_head = ProjectionHead(latent_dim) # rzutowanie cech do przestrzeni kontrastywnej


        # target branch -- teacher -- without gradients
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_projector_head = copy.deepcopy(self.online_projection_head)

        # stop gradients for the target branch
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        for p in self.target_projector_head.parameters():
            p.requires_grad = False


    def random_masking(self, x):
        # spradzić czy maskowanie jest ok, teraz jest na poziomioe pikseli/kanał a w oryginale jest maskowanie patchy
        B, C, H, W = x.shape
        mask = torch.rand(B, 1, H, W, device=x.device) > self.masking_ratio
        mask = mask.float()
        x_masked = x * mask

        return x_masked, mask
    # def random_masking(self, x):
    #     B, C, H, W = x.shape
    #     patch_size = 4  # Rozmiar bloku (dla 32x32 to daje siatkę 8x8)
        
    #     # robimy mała mask2 (np 8x8 )
    #     h_patches = H // patch_size
    #     w_patches = W // patch_size
        
    #     # losowanie na malej masce
    #     mask_small = torch.rand(B, 1, h_patches, w_patches, device=x.device) > self.masking_ratio
    #     mask_small = mask_small.float()
        
    #     # powiekszenie malej maski do 32X32

    #     mask = F.interpolate(mask_small, size=(H, W), mode='nearest')
        
    #     x_masked = x * mask
    #     return x_masked, mask

    def forward(self, x):
        x_masked, mask = self.random_masking(x)

        # online branch
        # encoder przetwarza zmaskowany obraz
        latent_online = self.online_encoder(x_masked)
        # pixel decoder
        reconstructed = self.pixel_decoder(latent_online)  # rekonstrukcja
        # feature decoder
        feature_online = self.feature_decoder(latent_online)
    

        projected_online = self.online_projection_head(feature_online) # projekcja do przestrzeni kontrastywnej aby porówanc z targetem

        # target branch
        with torch.no_grad():
            # encoder przetwarza oryginalny obraz
            latent_target = self.target_encoder(x)
            projected_target = self.target_projector_head(latent_target)


        return {
            'loss_recon': (x, reconstructed, mask), # Dane do straty rekonstrukcji
            'loss_contrast': (projected_online, projected_target), # Dane do straty kontrastywnej
            'reconstructed_image': reconstructed # Do wizualizacji
        }
    
    def update_target(self):
        # wzór: param_k = m * param_k + (1 - m) * param_q
        # param_k -- wagi target encoder i projector
        # param_q -- wagi online encoder i projector

        # aktualizacja wag target encoder i projector za pomocą momentum
        for param_q, param_k in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data = self.momentum * param_k.data + (1 - self.momentum) * param_q.data


        for param_q, param_k in zip(self.online_projection_head.parameters(), self.target_projector_head.parameters()):
            param_k.data = self.momentum * param_k.data + (1 - self.momentum) * param_q.data

    def compute_loss(self, outputs):
        # Strata rekonstrukcji

        original, reconstructed, mask = outputs['loss_recon']

        # MAE loss ---------------
        # masked_region = 1 - mask    
        # loss_recon = F.mse_loss(
        #     reconstructed * masked_region,
        #     original * masked_region,
        #     reduction='sum'
        # )

        # n_masked = masked_region.sum() # ile pikseli zamaskowanych
        # loss_recon = loss_recon / (n_masked + 1e-8) # srednia tylko z zmaskowanych

        # MAE loss lepsza wersja 
        masked_region = 1 - mask    
        loss_recon = F.mse_loss(
            reconstructed * masked_region,
            original * masked_region,
            reduction='mean'
        )

        # InfoNCE loss-------------Strata kontrastywna

        q, k = outputs['loss_contrast']

        # normalizacja wektorow cech
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)


        # liczenie podobienstwa -- kazdy obraz z studenta z kazdym z nauczyciela
        logits = torch.mm(q, k.t()) / self.temperature

        # labels = torch.arange(logits.size(0)).long().to(logits.device)
        labels = torch.arange(logits.size(0), device=logits.device)
        

        loss_contrast = F.cross_entropy(logits, labels)

        # laczna strata
        total_loss = loss_recon + (self.contrastive_loss_weight * loss_contrast)

        return total_loss, loss_recon, loss_contrast




