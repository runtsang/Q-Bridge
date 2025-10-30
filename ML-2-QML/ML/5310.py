import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any

class QuanvolutionFilter(nn.Module):
    """Convolutional filter inspired by the quanvolution example."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return features.view(x.size(0), -1)

class FullyConnectedLayer(nn.Module):
    """Simple fully‑connected layer to produce a scalar expectation."""
    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def run(self, thetas: np.ndarray) -> np.ndarray:
        values = torch.as_tensor(thetas, dtype=torch.float32).view(-1, 1)
        expectation = torch.tanh(self.linear(values)).mean(dim=0)
        return expectation.detach().numpy()

class SelfAttentionHybrid:
    """Hybrid self‑attention class that delegates similarity computation to a quantum module."""
    def __init__(self, embed_dim: int, quantum_attention: Any, use_quanvolution: bool = False):
        self.embed_dim = embed_dim
        self.quantum_attention = quantum_attention
        self.use_quanvolution = use_quanvolution
        if use_quanvolution:
            self.quanvolution = QuanvolutionFilter()
        self.fcl = FullyConnectedLayer(embed_dim * 14 * 14 if use_quanvolution else embed_dim)

    def run(self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            inputs: np.ndarray) -> np.ndarray:
        if self.use_quanvolution:
            inputs = self.quanvolution(torch.as_tensor(inputs, dtype=torch.float32)).detach().numpy()
        query = torch.as_tensor(inputs @ rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        key   = torch.as_tensor(inputs @ entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        value = torch.as_tensor(inputs, dtype=torch.float32)
        scores = torch.tensor(self.quantum_attention.run(rotation_params, entangle_params, query, key))
        weighted = torch.softmax(scores, dim=-1) @ value
        return self.fcl.run(weighted.numpy())
