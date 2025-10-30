import numpy as np
import torch
import torch.nn as nn

class SelfAttentionImpl(nn.Module):
    """
    Classical self‑attention module that first encodes 2‑D inputs with a lightweight CNN
    (inspired by Quantum‑NAT) and then applies a parametric attention block.
    The interface mirrors the original SelfAttention() function: run(rotation_params,
    entangle_params, inputs).
    """
    def __init__(self, embed_dim: int = 4, input_channels: int = 1, image_size: int = 28):
        super().__init__()
        self.embed_dim = embed_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        dummy = torch.zeros(1, input_channels, image_size, image_size)
        out = self.encoder(dummy)
        self.feature_dim = out.view(1, -1).shape[1]
        self.proj = nn.Linear(self.feature_dim, embed_dim)
        self.norm = nn.BatchNorm1d(embed_dim)

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(inputs.astype(np.float32))
        features = self.encoder(x)
        flattened = features.view(x.shape[0], -1)
        projected = self.proj(flattened)

        q = torch.from_numpy(rotation_params.reshape(self.embed_dim, -1)).float()
        k = torch.from_numpy(entangle_params.reshape(self.embed_dim, -1)).float()

        queries = torch.matmul(projected, q.T)
        keys = torch.matmul(projected, k.T)
        values = projected

        scores = torch.softmax(queries @ keys.T / np.sqrt(self.embed_dim), dim=-1)
        out = scores @ values
        out = self.norm(out)
        return out.detach().cpu().numpy()

def SelfAttention():
    """
    Factory that returns an instance of SelfAttentionImpl with default hyper‑parameters.
    """
    return SelfAttentionImpl(embed_dim=4)

__all__ = ["SelfAttentionImpl", "SelfAttention"]
