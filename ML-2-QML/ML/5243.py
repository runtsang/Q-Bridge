"""Hybrid classical-quantum neural network combining quanvolution, autoencoder and graph adjacency.

This module defines :class:`HybridQuanvolutionAutoGraphNet`, a purely classical network that emulates a quantum quanvolution filter, compresses the resulting features through a fully‑connected autoencoder, builds a similarity graph over the latent representations and produces class logits.  The architecture is deliberately modular so that each component can be swapped for a quantum counterpart while preserving the overall API.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# 1. Classical quanvolution filter – a lightweight 2×2 convolution
# --------------------------------------------------------------------------- #
class QuanvolutionFilter(nn.Module):
    """A simple 2×2 stride‑2 convolution that emulates a quantum kernel."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)  # shape (batch, 4, 14, 14)


# --------------------------------------------------------------------------- #
# 2. Autoencoder – compresses the flattened feature map
# --------------------------------------------------------------------------- #
class AutoencoderNet(nn.Module):
    """Multilayer perceptron autoencoder with configurable latent size."""
    def __init__(self, input_dim: int, latent_dim: int = 32,
                 hidden_dims: tuple[int, int] = (128, 64), dropout: float = 0.1) -> None:
        super().__init__()
        encoder = []
        in_dim = input_dim
        for h in hidden_dims:
            encoder += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout) if dropout > 0 else nn.Identity()]
            in_dim = h
        encoder += [nn.Linear(in_dim, latent_dim)]
        self.encoder = nn.Sequential(*encoder)

        decoder = []
        in_dim = latent_dim
        for h in reversed(hidden_dims):
            decoder += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout) if dropout > 0 else nn.Identity()]
            in_dim = h
        decoder += [nn.Linear(in_dim, input_dim)]
        self.decoder = nn.Sequential(*decoder)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


# --------------------------------------------------------------------------- #
# 3. Graph‑based similarity – adjacency from latent cosine similarity
# --------------------------------------------------------------------------- #
def compute_adjacency(latents: torch.Tensor, threshold: float = 0.8,
                     secondary: float | None = None, secondary_weight: float = 0.5) -> torch.Tensor:
    """
    Build a weighted adjacency matrix from pairwise cosine similarities.

    Args:
        latents: latent vectors of shape (batch, dim).
        threshold: primary similarity threshold.
        secondary: optional secondary threshold.
        secondary_weight: weight for secondary edges.

    Returns:
        adjacency matrix of shape (batch, batch) with values in [0, 1].
    """
    cos_sim = F.cosine_similarity(latents.unsqueeze(1), latents.unsqueeze(0), dim=2)
    adj = torch.zeros_like(cos_sim)
    adj[cos_sim >= threshold] = 1.0
    if secondary is not None:
        mask = (cos_sim >= secondary) & (cos_sim < threshold)
        adj[mask] = secondary_weight
    return adj


# --------------------------------------------------------------------------- #
# 4. Hybrid network
# --------------------------------------------------------------------------- #
class HybridQuanvolutionAutoGraphNet(nn.Module):
    """Classical hybrid network that fuses quanvolution, autoencoder and graph adjacency."""
    def __init__(self, latent_dim: int = 32, num_classes: int = 10) -> None:
        super().__init__()
        self.quanvolution = QuanvolutionFilter()
        # feature map size after conv: 4 × 14 × 14 = 784
        self.autoencoder = AutoencoderNet(input_dim=4 * 14 * 14, latent_dim=latent_dim)
        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Classical quanvolution
        features = self.quanvolution(x)                     # (B, 4, 14, 14)
        flat = features.view(features.size(0), -1)          # (B, 784)

        # 2. Autoencoder latent representation
        latent = self.autoencoder.encode(flat)              # (B, latent_dim)

        # 3. Graph refinement
        adj = compute_adjacency(latent)
        # Normalise rows to keep probabilities
        row_sums = adj.sum(dim=1, keepdim=True).clamp_min_(1e-6)
        refined = latent @ (adj / row_sums)

        # 4. Classification
        logits = self.classifier(refined)
        return F.log_softmax(logits, dim=-1)

__all__ = ["HybridQuanvolutionAutoGraphNet", "QuanvolutionFilter", "AutoencoderNet"]
