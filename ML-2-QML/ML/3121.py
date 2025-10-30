"""Hybrid quanvolution autoencoder combining quantum filter with classical decoder."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq

class QuanvolutionFilter(tq.QuantumModule):
    """Apply a random two‑qubit quantum kernel to 2×2 image patches."""
    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                data = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, data)
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)

class QuanvolutionAutoencoder(nn.Module):
    """Hybrid encoder–decoder: quantum quanvolution encoder + classical MLP decoder."""
    def __init__(self, latent_dim: int = 32):
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(4 * 14 * 14, 512),
            nn.ReLU(),
            nn.Linear(512, self.latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 4 * 14 * 14),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        latent = self.encoder(features)
        reconstruction = self.decoder(latent)
        return reconstruction.view(x.size(0), 1, 28, 28)

def QuanvolutionAutoencoderFactory(latent_dim: int = 32) -> QuanvolutionAutoencoder:
    return QuanvolutionAutoencoder(latent_dim)

__all__ = ["QuanvolutionFilter", "QuanvolutionAutoencoder", "QuanvolutionAutoencoderFactory"]
