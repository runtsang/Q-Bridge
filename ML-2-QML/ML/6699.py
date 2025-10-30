import torch
import torch.nn as nn
import numpy as np

class SelfAttention(nn.Module):
    """Classical self‑attention augmented with a learnable 2‑D convolutional front‑end."""

    def __init__(self, embed_dim: int, kernel_size: int = 2, threshold: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        self.threshold = threshold

        # Convolutional feature extractor (drop‑in for a quanvolution)
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)
        self.sigmoid = nn.Sigmoid()

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        """
        Args:
            rotation_params: 1‑D array of length embed_dim*embed_dim, reshaped to (embed_dim, embed_dim).
            entangle_params: 1‑D array of length embed_dim*embed_dim, reshaped to (embed_dim, embed_dim).
            inputs: 2‑D array with shape (kernel_size, kernel_size) for convolution followed by
                    (batch, seq_len, embed_dim) for attention.

        Returns:
            1‑D numpy array of the attended representation.
        """
        # Convolutional pre‑processing
        data = torch.as_tensor(inputs, dtype=torch.float32)
        if data.ndim == 2:
            data = data.view(1, 1, self.kernel_size, self.kernel_size)
            conv_out = self.conv(data)
            conv_out = self.sigmoid(conv_out - self.threshold)
            conv_out = conv_out.mean().item()
            return np.array([conv_out], dtype=np.float32)

        # Multi‑head attention
        query = torch.as_tensor(inputs @ rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        key   = torch.as_tensor(inputs @ entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        value = torch.as_tensor(inputs, dtype=torch.float32)

        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return (scores @ value).detach().numpy()

__all__ = ["SelfAttention"]
