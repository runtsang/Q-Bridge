import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple

@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1

class HybridQCNNAutoEncoder(nn.Module):
    """
    Classical backbone that combines a 2‑D convolution filter,
    a fully‑connected auto‑encoder, and a lightweight classifier.
    """
    def __init__(self,
                 input_dim: int,
                 latent_dim: int = 32,
                 hidden_dims: Tuple[int, int] = (128, 64),
                 dropout: float = 0.1,
                 conv_kernel: int = 2,
                 conv_threshold: float = 0.0) -> None:
        super().__init__()
        # 2‑D convolution filter
        self.conv_filter = nn.Conv2d(1, 1, kernel_size=conv_kernel, bias=True)
        self.conv_threshold = conv_threshold
        # Auto‑encoder encoder
        self.encoder = nn.Sequential(
            *self._build_mlp(input_dim, hidden_dims, latent_dim, dropout, encoder=True)
        )
        # Auto‑encoder decoder
        self.decoder = nn.Sequential(
            *self._build_mlp(latent_dim, hidden_dims, input_dim, dropout, encoder=False)
        )
        # Classifier on latent space
        self.classifier = nn.Linear(latent_dim, 1)

    def _build_mlp(self,
                   in_dim: int,
                   hidden_dims: Tuple[int, int],
                   out_dim: int,
                   dropout: float,
                   encoder: bool) -> Tuple[nn.Module]:
        layers = []
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, out_dim))
        return tuple(layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # If input is 2‑D image (batch, 1, H, W) we apply conv filter
        if x.dim() == 4:
            x = self.conv_filter(x)
            x = torch.sigmoid(x - self.conv_threshold)
            x = x.view(x.size(0), -1)
        # Encode
        latent = self.encoder(x)
        # Decode (reconstruction)
        recon = self.decoder(latent)
        # Classification logits
        logits = self.classifier(latent)
        return torch.sigmoid(logits), recon
