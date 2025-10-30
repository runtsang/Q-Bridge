from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx

from GraphQNN import feedforward, fidelity_adjacency, random_network
from SelfAttention import SelfAttention
from FraudDetection import build_fraud_detection_program, FraudLayerParameters

class QuantumNATGen200(nn.Module):
    """Hybrid classical model that fuses CNN feature extraction, graph neural propagation,
    self‑attention, and fraud‑detection style parameterized layers."""
    def __init__(self, in_channels: int = 1, num_classes: int = 1):
        super().__init__()
        # 1. Convolutional backbone
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        # 2. Graph neural network parameters
        self.gnn_arch = [32, 32, 16]
        _, self.gnn_weights, _, _ = random_network(self.gnn_arch, samples=10)
        # 3. Self‑attention module
        self.attention = SelfAttention()
        # 4. Fraud‑detection style feed‑forward
        self.fraud_prog = build_fraud_detection_program(
            FraudLayerParameters(
                bs_theta=0.0, bs_phi=0.0,
                phases=(0.0, 0.0),
                squeeze_r=(0.0, 0.0),
                squeeze_phi=(0.0, 0.0),
                displacement_r=(0.0, 0.0),
                displacement_phi=(0.0, 0.0),
                kerr=(0.0, 0.0)
            ),
            [
                FraudLayerParameters(
                    bs_theta=0.0, bs_phi=0.0,
                    phases=(0.0, 0.0),
                    squeeze_r=(0.0, 0.0),
                    squeeze_phi=(0.0, 0.0),
                    displacement_r=(0.0, 0.0),
                    displacement_phi=(0.0, 0.0),
                    kerr=(0.0, 0.0)
                )
            ]
        )
        # 5. Dimensionality reduction before fraud module
        self.reduce = nn.Linear(32, 2)
        # 6. Final classifier
        self.classifier = nn.Linear(2, num_classes)
        self.bn = nn.BatchNorm1d(2)

        # Parameters for self‑attention (fixed across batches)
        self.rotation_params = nn.Parameter(torch.randn(4, 4))
        self.entangle_params = nn.Parameter(torch.randn(3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feature extraction
        feats = self.features(x)          # (B, 32, 1, 1)
        feats = feats.view(feats.size(0), -1)  # (B, 32)

        # Graph neural network propagation
        samples = [(feats[i], feats[i]) for i in range(feats.size(0))]
        gnn_out = feedforward(self.gnn_arch, self.gnn_weights, samples)
        # Use the last layer activations as node representations
        gnn_repr = torch.stack([activations[-1] for activations in gnn_out], dim=0)  # (B, 16)

        # Self‑attention on aggregated representation
        rotation_np = self.rotation_params.detach().cpu().numpy()
        entangle_np = self.entangle_params.detach().cpu().numpy()
        attn_np = self.attention.run(
            rotation_params=rotation_np,
            entangle_params=entangle_np,
            inputs=gnn_repr.detach().cpu().numpy()
        )
        attn_repr = torch.tensor(attn_np, dtype=torch.float32, device=x.device)

        # Reduce dimensionality for fraud‑detection block
        reduced = self.reduce(attn_repr)  # (B, 2)

        # Fraud‑detection style processing
        fraud_out = self.fraud_prog(reduced)  # (B, 1)

        # Final classification
        out = self.classifier(fraud_out)
        out = self.bn(out)
        return out

__all__ = ["QuantumNATGen200"]
