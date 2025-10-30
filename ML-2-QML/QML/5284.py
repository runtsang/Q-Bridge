"""Quantum‑enhanced vision classifier.

The HybridVisionClassifier below replaces the classical quanvolution filter
with a variational two‑qubit kernel and the sampler with a Qiskit SamplerQNN.
All other components (CNN backbone, transformer block) remain classical,
allowing end‑to‑end training with a hybrid quantum‑classical loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN as QiskitSamplerQNN
from qiskit.primitives import StatevectorSampler as Sampler

# Quantum quanvolution filter (variational 2‑qubit kernel)
class QuantumQuanvolutionFilter(tq.QuantumModule):
    def __init__(self):
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

# Classical transformer components (same as the ML version)
class MultiHeadAttentionClassical(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_output, _ = self.attn(x, x, x, key_padding_mask=mask)
        return attn_output


class FeedForwardClassical(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerBlockClassical(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

# Quantum sampler using Qiskit
def QuantumSamplerQNN() -> QiskitSamplerQNN:
    inputs = ParameterVector("input", 2)
    weights = ParameterVector("weight", 4)
    qc = QuantumCircuit(2)
    qc.ry(inputs[0], 0)
    qc.ry(inputs[1], 1)
    qc.cx(0, 1)
    qc.ry(weights[0], 0)
    qc.ry(weights[1], 1)
    qc.cx(0, 1)
    qc.ry(weights[2], 0)
    qc.ry(weights[3], 1)
    sampler = Sampler()
    return QiskitSamplerQNN(circuit=qc, input_params=inputs, weight_params=weights, sampler=sampler)

# Main hybrid classifier
class HybridVisionClassifier(tq.QuantumModule):
    """
    Quantum‑enhanced vision classifier that fuses a classical CNN backbone,
    a variational quanvolution filter, a classical transformer block,
    and a Qiskit SamplerQNN before producing logits.
    """
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        # Classical backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Quantum quanvolution
        self.quanv = QuantumQuanvolutionFilter()
        # Classical transformer
        self.transformer = TransformerBlockClassical(embed_dim=16, num_heads=4, ffn_dim=32)
        # Quantum sampler
        self.sampler = QuantumSamplerQNN()
        # Feature projector to 2D for sampler
        self.feature_proj = nn.Linear(16 * 14 * 14 + 4 * 14 * 14, 2)
        # Final classifier
        self.classifier = nn.Linear(2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Backbone
        x_backbone = self.backbone(x)                     # (B, 16, H/2, W/2)
        # Flatten for transformer
        B, C, H, W = x_backbone.shape
        patches = x_backbone.view(B, C, -1).transpose(1, 2)  # (B, H*W, C)
        # Transformer
        x_trans = self.transformer(patches)                # (B, H*W, C)
        x_trans = x_trans.mean(dim=1)                      # (B, C)
        # Quantum quanvolution on grayscale projection
        gray = x.mean(dim=1, keepdim=True)                 # (B, 1, H, W)
        q_feat = self.quanv(gray)                          # (B, 4*H/2*W/2)
        # Combine features
        combined = torch.cat((x_trans, q_feat), dim=1)      # (B, 16*14*14 + 4*14*14)
        proj = self.feature_proj(combined)                  # (B, 2)
        sampled = self.sampler(proj)                        # (B, 2)
        logits = self.classifier(sampled)                   # (B, num_classes)
        return logits


__all__ = ["HybridVisionClassifier"]
