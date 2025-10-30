import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

class ClassicalSelfAttention(nn.Module):
    """A lightweight self‑attention block with trainable query/key projections."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.query_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        query = self.query_proj(inputs)
        key = self.key_proj(inputs)
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return scores @ inputs

class QModel(nn.Module):
    """Hybrid classical regression model: self‑attention feature extractor + linear head."""
    def __init__(self, num_features: int):
        super().__init__()
        self.attention = ClassicalSelfAttention(num_features)
        self.head = nn.Linear(num_features, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        attn_features = self.attention(state_batch)
        return self.head(attn_features).squeeze(-1)

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
