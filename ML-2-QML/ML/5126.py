"""Hybrid classical sampler/classifier/regressor model."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

class ClassicalSelfAttention:
    """Simple self‑attention block mirroring the quantum interface."""
    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        query = torch.as_tensor(inputs @ rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        key = torch.as_tensor(inputs @ entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        value = torch.as_tensor(inputs, dtype=torch.float32)
        scores = F.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return (scores @ value).numpy()

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample states of the form cos(theta)|0..0> + e^{i phi} sin(theta)|1..1>."""
    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    y = np.sin(2 * thetas) * np.cos(phis)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset for regression tasks."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict:
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class HybridSamplerModel(nn.Module):
    """Classical hybrid model supporting sampling, classification and regression."""
    def __init__(self, mode: str, num_features: int, depth: int = 1, use_attention: bool = False):
        super().__init__()
        self.mode = mode.lower()
        self.encoder = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.Tanh(),
        )
        self.use_attention = use_attention
        if use_attention:
            self.attention = ClassicalSelfAttention(embed_dim=num_features)
        self.hidden_layers = nn.ModuleList()
        for _ in range(depth):
            self.hidden_layers.append(nn.Linear(num_features, num_features))
            self.hidden_layers.append(nn.ReLU())
        if self.mode in ("sampler", "classifier"):
            self.head = nn.Linear(num_features, 2)
        elif self.mode == "regressor":
            self.head = nn.Linear(num_features, 1)
        else:
            raise ValueError(f"Unsupported mode {mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.encoder(x)
        if self.use_attention:
            rot = np.random.randn(self.encoder[0].out_features, self.encoder[0].out_features)
            ent = np.random.randn(self.encoder[0].out_features, self.encoder[0].out_features)
            out = torch.from_numpy(self.attention.run(rot, ent, out.detach().cpu().numpy())).to(out.device)
        for layer in self.hidden_layers:
            out = layer(out)
        out = self.head(out)
        if self.mode in ("sampler", "classifier"):
            return F.softmax(out, dim=-1)
        return out.squeeze(-1)

__all__ = ["HybridSamplerModel", "RegressionDataset", "generate_superposition_data"]
