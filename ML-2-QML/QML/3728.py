"""
Hybrid quantum autoencoder that maps CNN‑extracted features into a 4‑wire quantum
state, applies a variational circuit, measures to obtain a latent vector, and
reconstructs the image with a classical MLP.  The design is built on the
`QuantumNAT.py` quantum module and the quantum autoencoder from
`Autoencoder.py`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from torchquantum import encoder_op_list_name_dict


class HybridNATAutoencoder(tq.QuantumModule):
    """Quantum‑classical autoencoder with a 4‑wire latent space."""

    class QLayer(tq.QuantumModule):
        """Variational layer that expands the entanglement of the 4‑wire system."""

        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)
            self.crx = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            self.rx(qdev, wires=0)
            self.ry(qdev, wires=1)
            self.rz(qdev, wires=2)
            self.crx(qdev, wires=[0, 2])
            tqf.hadamard(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
            tqf.sx(qdev, wires=1, static=self.static_mode, parent_graph=self.graph)
            tqf.cnot(qdev, wires=[2, 0], static=self.static_mode, parent_graph=self.graph)

    def __init__(self, latent_dim: int = 4, hidden_dims=(128, 64), dropout: float = 0.1):
        super().__init__()
        self.n_wires = latent_dim
        self.encoder = tq.GeneralEncoder(encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer(self.n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

        # Classical decoder MLP that mirrors the hybrid autoencoder
        dec_layers = []
        in_dim = self.n_wires
        for h in reversed(hidden_dims):
            dec_layers.append(nn.Linear(in_dim, h))
            dec_layers.append(nn.ReLU())
            if dropout > 0.0:
                dec_layers.append(nn.Dropout(dropout))
            in_dim = h
        dec_layers.append(nn.Linear(in_dim, 16 * 7 * 7))
        self.decoder = nn.Sequential(*dec_layers)

        # Reconstruction head – de‑convolutional layers
        self.recon_head = nn.Sequential(
            nn.Unflatten(1, (16, 7, 7)),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode, quantum‑transform, and decode ``x``."""
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)

        # Classical pooling to match the 4‑wire encoder
        pooled = F.avg_pool2d(x, 6).view(bsz, -1)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        latent = self.measure(qdev)
        latent = self.norm(latent)

        feats = self.decoder(latent)
        return self.recon_head(feats)


__all__ = ["HybridNATAutoencoder"]
