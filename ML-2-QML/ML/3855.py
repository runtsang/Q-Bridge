import numpy as np
import torch
import torch.nn as nn

class ClassicalSelfAttention:
    """Classical self‑attention that mirrors the quantum interface."""
    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        query = torch.as_tensor(inputs @ rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        key   = torch.as_tensor(inputs @ entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        value = torch.as_tensor(inputs, dtype=torch.float32)
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return (scores @ value).numpy()

class HybridSelfAttention(nn.Module):
    """Classical self‑attention + CNN inspired by Quantum‑NAT."""
    def __init__(self):
        super().__init__()
        self.attention = ClassicalSelfAttention(embed_dim=4)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor, rotation_params: np.ndarray, entangle_params: np.ndarray) -> torch.Tensor:
        # Extract features with the CNN
        features = self.cnn(x)
        flattened = features.view(features.shape[0], -1)
        # Apply classical attention on the flattened features
        attn_out = torch.from_numpy(self.attention.run(rotation_params, entangle_params, flattened.cpu().numpy()))
        # Final fully‑connected projection
        out = self.fc(attn_out)
        return self.norm(out)

__all__ = ["HybridSelfAttention"]
