import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic regression dataset with a nonlinear target.
    The target is a combination of sin and cos functions of the sum of input features.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """
    Dataset that returns raw feature vectors and target scalars.
    """
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class SimpleTransformerEncoder(nn.Module):
    """
    A minimal transformer encoder block that operates purely classically.
    """
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x

class UnifiedRegressionTransformer(nn.Module):
    """
    Classical regression model that combines a transformer encoder with a simulated quantum feature extractor.
    """
    def __init__(
        self,
        num_features: int,
        embed_dim: int = 32,
        num_heads: int = 4,
        ffn_dim: int = 64,
        quantum_dim: int = 8,
    ):
        super().__init__()
        self.input_proj = nn.Linear(num_features, embed_dim)
        self.classical_encoder = SimpleTransformerEncoder(embed_dim, num_heads, ffn_dim)
        # Simulated quantum feature extractor (purely classical network)
        self.quantum_encoder = nn.Sequential(
            nn.Linear(num_features, quantum_dim),
            nn.ReLU(),
            nn.Linear(quantum_dim, quantum_dim),
        )
        self.head = nn.Linear(embed_dim + quantum_dim, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        # Classical path
        token = self.input_proj(state_batch)  # [B, E]
        token = token.unsqueeze(1)  # [B, 1, E]
        token = self.classical_encoder(token)  # [B, 1, E]

        # Simulated quantum path
        q_features = self.quantum_encoder(state_batch)  # [B, Q]

        # Concatenate and produce output
        combined = torch.cat([token.squeeze(1), q_features], dim=-1)
        return self.head(combined).squeeze(-1)

__all__ = ["UnifiedRegressionTransformer", "RegressionDataset", "generate_superposition_data"]
