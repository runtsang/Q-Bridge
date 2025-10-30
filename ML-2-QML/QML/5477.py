import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
import math

class QuantumPatchEncoder(tq.QuantumModule):
    """Quantum 2×2 patch extractor that maps pixel values to a 4‑dimensional output."""
    def __init__(self, n_wires: int = 4) -> None:
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
        self.random_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, patch)
                self.random_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)  # (B, 196*4)

class QuantumAutoEncoder(tq.QuantumModule):
    """Variational auto‑encoder that compresses 4‑dimensional patch features."""
    def __init__(self, latent_dim: int = 4) -> None:
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.random_layer = tq.RandomLayer(n_ops=10, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.project = nn.Linear(self.n_wires, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, seq_len, 4)
        bsz, seq_len, _ = x.shape
        device = x.device
        latents = []
        for i in range(seq_len):
            patch = x[:, i, :]  # (B, 4)
            qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
            self.encoder(qdev, patch)
            self.random_layer(qdev)
            measurement = self.measure(qdev)
            latent = self.project(measurement)
            latents.append(latent)
        latents = torch.stack(latents, dim=1)  # (B, seq_len, latent_dim)
        return latents

class ClassicalTransformerClassifier(nn.Module):
    """Transformer classifier that processes the latent sequence."""
    def __init__(self, seq_len: int, embed_dim: int, num_heads: int,
                 num_layers: int, num_classes: int) -> None:
        super().__init__()
        self.positional = nn.Parameter(torch.randn(1, seq_len, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.positional
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x)

class QuanvolutionAutoTransformer(tq.QuantumModule):
    """Quantum‑enhanced version of the hybrid architecture."""
    def __init__(self, latent_dim: int = 4, num_classes: int = 10) -> None:
        super().__init__()
        self.patch_encoder = QuantumPatchEncoder()
        self.autoencoder = QuantumAutoEncoder(latent_dim=latent_dim)
        self.transformer = ClassicalTransformerClassifier(
            seq_len=14 * 14,
            embed_dim=latent_dim,
            num_heads=4,
            num_layers=2,
            num_classes=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encode patches with quantum circuit
        patches = self.patch_encoder(x)          # (B, 196*4)
        patches = patches.view(x.size(0), 14 * 14, 4)  # (B, 196, 4)
        # Compress with quantum auto‑encoder
        latent = self.autoencoder(patches)       # (B, 196, latent_dim)
        # Classify with transformer
        logits = self.transformer(latent)
        return logits

__all__ = ["QuanvolutionAutoTransformer"]
