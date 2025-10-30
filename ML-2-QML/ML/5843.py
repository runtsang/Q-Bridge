"""Hybrid classical‑quantum model combining CNN, autoencoder, and a variational quantum layer."""

from __future__ import annotations

import torch
import torch.nn as nn
import torchquantum as tq
from dataclasses import dataclass
from typing import Tuple

@dataclass
class HybridConfig:
    """Configuration for the hybrid QuantumNAT model."""
    input_channels: int = 1
    feature_maps: Tuple[int, int] = (8, 16)
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    output_dim: int = 4

class QFCModel(nn.Module):
    """Classical CNN + autoencoder + quantum latent processing to 4 outputs."""

    def __init__(self, config: HybridConfig = HybridConfig()):
        super().__init__()
        # Convolutional feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(config.input_channels, config.feature_maps[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(config.feature_maps[0], config.feature_maps[1], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Dummy forward to compute feature map size
        dummy = torch.zeros(1, config.input_channels, 28, 28)
        feat_size = self.features(dummy).view(1, -1).size(1)

        # Classical encoder to latent vector
        self.encoder = nn.Sequential(
            nn.Linear(feat_size, config.hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(config.hidden_dims[0], config.latent_dim),
        )

        # Quantum latent processing
        self.q_layer = QuantumLayer(n_wires=4, n_ops=50)

        # Classical decoder back to 4 outputs
        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(config.hidden_dims[1], config.output_dim),
        )
        self.norm = nn.BatchNorm1d(config.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        feat = self.features(x)
        flat = feat.view(bsz, -1)
        latent = self.encoder(flat)
        q_latent = self.q_layer(latent)
        out = self.decoder(q_latent)
        return self.norm(out)

class QuantumLayer(tq.QuantumModule):
    """Variational quantum block that transforms a latent vector."""

    def __init__(self, n_wires: int = 4, n_ops: int = 50):
        super().__init__()
        self.n_wires = n_wires
        # Encoder maps classical angles to quantum gates
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.random_layer = tq.RandomLayer(n_ops=n_ops, wires=list(range(n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(n_wires)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        bsz = latent.shape[0]
        angles = latent[:, :self.n_wires]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=latent.device, record_op=True)
        self.encoder(qdev, angles)
        self.random_layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)

__all__ = ["QFCModel"]
