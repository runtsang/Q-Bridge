import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic regression dataset."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int):
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class ClassicalSelfAttention(nn.Module):
    """Simple self‑attention block."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj   = nn.Linear(embed_dim, embed_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        query = self.query_proj(inputs)
        key   = self.key_proj(inputs)
        scores = F.softmax(query @ key.transpose(-1, -2) / np.sqrt(self.embed_dim), dim=-1)
        return scores @ inputs

class QuanvolutionFilter(nn.Module):
    """Classical 2‑D convolution that mimics a quantum kernel."""
    def __init__(self, in_channels: int = 1, out_channels: int = 4, kernel_size: int = 2, stride: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return features.view(x.size(0), -1)

class HybridRegression(nn.Module):
    """Hybrid regression model that chains self‑attention, quanvolution and a linear head."""
    def __init__(self, num_features: int, embed_dim: int = 4, conv_out_channels: int = 4):
        super().__init__()
        self.attention = ClassicalSelfAttention(embed_dim=embed_dim)
        self.quanvolution = QuanvolutionFilter(in_channels=1, out_channels=conv_out_channels)
        self.linear = nn.Linear(embed_dim * conv_out_channels, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        # Classical self‑attention
        attn_out = self.attention(state_batch)
        # Reshape for 2‑D convolution (square feature size)
        H = W = int(np.sqrt(state_batch.shape[1]))
        if H * W!= state_batch.shape[1]:
            pad = H * W - state_batch.shape[1]
            attn_out = F.pad(attn_out, (0, pad))
        attn_out_2d = attn_out.view(state_batch.shape[0], 1, H, W)
        # Quanvolution
        quanv_out = self.quanvolution(attn_out_2d)
        # Linear head
        return self.linear(quanv_out).squeeze(-1)

__all__ = ["HybridRegression", "RegressionDataset", "generate_superposition_data"]
