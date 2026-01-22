import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from src.models.cmae_decoders import PixelDecoder, FeatureDecoder
from src.models.encoder import Encoder
import torchvision.transforms as T


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


class Predictor(nn.Module):
    def __init__(self, dim=128, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        return self.net(x)


class CMAE(nn.Module):

    def __init__(
            self,
            latent_dim=256,
            masking_ratio=0.75,
            # momentum=0.996, # parameter for updating the target encoder
            momentum = 0.99,
            temperature=0.07, # temperature for contrastive loss
            contrastive_loss_weight=1, # weight for contrastive loss
    ):
        
        super().__init__()

        self.masking_ratio = masking_ratio
        self.momentum = momentum
        self.temperature = temperature
        self.contrastive_loss_weight = contrastive_loss_weight


        # ------ online branch -- student-- with gradients
        self.online_encoder = Encoder(latent_dim)
        self.pixel_decoder = PixelDecoder(latent_dim)       # rekonstrukcja pikseli
        self.feature_decoder = FeatureDecoder(latent_dim)   # ekstrakcja cech
        # projector + predictor
        self.online_projection_head = ProjectionHead(latent_dim) # rzutowanie cech do przestrzeni kontrastywnej
        self.online_predictor = Predictor(dim=128, hidden_dim=256)


        # target branch -- teacher -- without gradients
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_projector_head = copy.deepcopy(self.online_projection_head)


        # stop gradients for the target branch
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        for p in self.target_projector_head.parameters():
            p.requires_grad = False


    def forward(self, x):

        # --- pixel shifting
        # view_online, view_target = self.pixel_shift_transform(x)
        # x_masked, mask = self.random_masking(view_online)


        # --- bez pixel shifting
        x_masked, mask = self.random_masking(x)
        view_target = x
        view_online = x


        # -----------online branch
        latent_online = self.online_encoder(x_masked)
        reconstructed = self.pixel_decoder(latent_online)

        # feature decoder -- kontrastywne
        feature_online = self.feature_decoder(latent_online)
        projected_online = self.online_projection_head(feature_online)  # projekcja do przestrzeni kontrastywnej aby porówanc z targetem
        predicted = self.online_predictor(projected_online)

        # target branch
        with torch.no_grad():
            # encoder przetwarza oryginalny obraz
            latent_target = self.target_encoder(view_target)
            projected_target = self.target_projector_head(latent_target)


        return {
            'loss_recon': (view_online, reconstructed, mask),  # Dane do straty rekonstrukcji
            # 'loss_contrast': (projected_online, projected_target), # Dane do straty kontrastywnej bez predictora
            'loss_contrast': (predicted, projected_target),  # Dane do straty kontrastywnej
            'reconstructed_image': reconstructed  # Do wizualizacji
        }

    # def random_masking(self, x):  # piksele
    #     B, C, H, W = x.shape
    #     mask = torch.rand(B, 1, H, W, device=x.device) > self.masking_ratio
    #     mask = mask.float()
    #     x_masked = x * mask
    #
    #     return x_masked, mask

    def random_masking(self, x):  # patche
        """
        Maskowanie losowe na poziomie patchy.
        Args:
            x: Tensor obrazu  (B, C, H, W)
        Returns:
            x_masked: zamaskowany obraz (B, C, H, W)
            mask: Tensor maski (B, 1, H, W)  1=widoczny, 0=zamaskowany
        """
        B, C, H, W = x.shape
        patch_size = 4 # (32x32 -> 8x8 grid)

        # wymiar siatki patchy
        h_patches = H // patch_size
        w_patches = W // patch_size

        mask_patches = torch.rand(B, h_patches, w_patches, device=x.device) > self.masking_ratio
        mask_patches = mask_patches.float()

        mask = F.interpolate(mask_patches.unsqueeze(1), size=(H, W), mode='nearest')

        x_masked = x * mask

        return x_masked, mask

    def pixel_shift_transform(self, x, shift_range=2):
        """
        Implementacja pixel shifting

        Args:
            x: Input image [B, C, H, W]
            shift_range: Maximum pixel shift

        Returns:
            view_online: View for online encoder
            view_target: View for momentum encoder
        """
        B, C, H, W = x.shape
        p = shift_range
        # dodanie paddingu
        x_pad = F.pad(x, (p, p, p, p), mode='reflect')
        # przesuniecie o r pikseli dla target view
        rh = torch.randint(0, p + 1, (1,)).item()
        rw = torch.randint(0, p + 1, (1,)).item()

        view_online = x_pad[:, :, p:p+H, p:p+W]
        view_target = x_pad[:, :, rh:rh+H, rw:rw+W]


        return view_online, view_target


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
        B, C, H, W = original.shape

        # ------------loss dla pixeli

        # original_norm = original
        #
        #
        # # -----------------------loss dla patchy


        # normalizacja targetu per patch
        patch_size = 4
        patches = original.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)

        mean = patches.mean(dim=[1, 4, 5], keepdim=True)
        var = patches.var(dim=[1, 4, 5], keepdim=True, unbiased=True)

        patches_norm = (patches - mean) / (var + 1e-6).sqrt()

        # rekonstrukcja patchy
        num_patches_h = H // patch_size
        num_patches_w = W // patch_size
        original_norm = patches_norm.permute(0, 1, 2, 4, 3, 5).contiguous()
        original_norm = original_norm.view(B, C, H, W)
        # ---------------------------------------

        masked_region = 1 - mask
        loss_recon = F.mse_loss(
            reconstructed * masked_region,
            original_norm * masked_region,
            reduction='sum'
        )
        n_masked = masked_region.sum() * C # ile pikseli zamaskowanych
        loss_recon = loss_recon / (n_masked + 1e-8)  # srednia tylko z zmaskowanych


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

