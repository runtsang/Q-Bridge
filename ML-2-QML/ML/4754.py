"""Hybrid classical architecture combining CNN, autoencoder, and QCNN‑style fully connected layers."""
from __future__ import annotations

import torch
from torch import nn

class HybridCNNAutoModel(nn.Module):
    """
    Classical hybrid network that:
      1. Extracts features via a small CNN.
      2. Projects to a latent space using a fully connected block.
      3. Compresses and reconstructs the latent vector with a dense autoencoder.
      4. Passes the reconstructed latent through QCNN‑style linear layers.
      5. Normalises the final 4‑dimensional output.
    """
    def __init__(self) -> None:
        super().__init__()
        # 1. CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # 2. Fully connected projection to 64‑dim latent
        self.fc_proj = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        # 3. Autoencoder for 32‑dim latent
        self.autoencoder = self._build_autoencoder(in_dim=32, latent_dim=16)
        # 4. QCNN‑style linear layers emulating quantum pooling
        self.qcnn_layers = nn.Sequential(
            nn.Linear(16, 24), nn.Tanh(),
            nn.Linear(24, 24), nn.Tanh(),
            nn.Linear(24, 20), nn.Tanh(),
            nn.Linear(20, 12), nn.Tanh(),
            nn.Linear(12, 8),  nn.Tanh(),
            nn.Linear(8, 4),   nn.Tanh()
        )
        # 5. Output normalisation
        self.norm = nn.BatchNorm1d(4)

    @staticmethod
    def _build_autoencoder(in_dim: int, latent_dim: int) -> nn.Module:
        """Return a symmetric autoencoder with one hidden layer."""
        encoder = nn.Sequential(
            nn.Linear(in_dim, latent_dim),
            nn.ReLU()
        )
        decoder = nn.Sequential(
            nn.Linear(latent_dim, in_dim),
            nn.ReLU()
        )
        return nn.ModuleDict({"enc": encoder, "dec": decoder})

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # CNN feature extraction
        feats = self.cnn(x)
        flat = feats.view(x.size(0), -1)
        # FC projection
        latent = self.fc_proj(flat)
        # Autoencoder compression & reconstruction
        compressed = self.autoencoder["enc"](latent)
        reconstructed = self.autoencoder["dec"](compressed)
        # QCNN‑style linear emulation
        out = self.qcnn_layers(reconstructed)
        return self.norm(out)

__all__ = ["HybridCNNAutoModel"]
