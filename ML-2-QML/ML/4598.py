from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantumInspiredLayer(nn.Module):
    """A classical surrogate for a quantum kernel using sine activations."""
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.linear2 = nn.Linear(out_features, out_features)
        self.linear3 = nn.Linear(out_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = torch.sin(x)
        x = self.linear2(x)
        x = torch.sin(x)
        x = self.linear3(x)
        return x


class HybridNATAutoEncoder(nn.Module):
    """
    Classical hybrid model that mirrors the structure of the original QuantumNAT
    while adding a quantum‑inspired non‑linear layer and an auto‑encoder head.
    """
    def __init__(self) -> None:
        super().__init__()
        # Feature extractor – identical to QFCModel's CNN backbone
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Projection to the 4‑dimensional NAT output
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )
        self.norm = nn.BatchNorm1d(4)

        # Quantum‑inspired non‑linear block
        self.quantum = QuantumInspiredLayer(64, 64)

        # Auto‑encoder decoder
        self.decoder = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 16 * 7 * 7),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            latent: 4‑dimensional NAT‑style output
            recon: 28×28 reconstructed image (same shape as input)
        """
        bsz = x.shape[0]
        f = self.features(x)
        flat = f.view(bsz, -1)
        latent = self.fc(flat)
        latent = self.norm(latent)

        # Quantum‑inspired processing
        q_out = self.quantum(latent)

        # Reconstruction through the decoder
        recon_flat = self.decoder(q_out)
        recon = recon_flat.view(bsz, 1, 28, 28)

        return latent, recon


__all__ = ["HybridNATAutoEncoder"]
