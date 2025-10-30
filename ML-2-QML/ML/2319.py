import numpy as np
import torch
from torch import nn

class HybridSelfAttention:
    """
    Classical hybrid self‑attention module.
    Applies a 2×2 convolution to each input vector reshaped as a 2‑D patch,
    then performs scaled dot‑product attention across the sequence.
    """
    def __init__(self, embed_dim: int, kernel_size: int = 2, threshold: float = 0.0):
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        self.threshold = threshold
        # Convolutional filter
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def _conv_feature(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply convolution to each element of the sequence.
        x: (batch, seq_len, embed_dim)
        Returns: (batch, seq_len, 1)
        """
        batch, seq_len, _ = x.shape
        # reshape to (batch*seq_len, 1, kernel, kernel)
        x_reshaped = x.view(batch * seq_len, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(x_reshaped)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.view(batch, seq_len, 1)

    def run(self, inputs: np.ndarray) -> np.ndarray:
        """
        Compute hybrid self‑attention.
        inputs: (batch, seq_len, embed_dim)
        Returns: (batch, seq_len, seq_len) attention matrix.
        """
        x = torch.as_tensor(inputs, dtype=torch.float32)
        conv_feat = self._conv_feature(x)
        query = torch.matmul(x, torch.eye(self.embed_dim, dtype=torch.float32))
        key   = torch.matmul(x, torch.eye(self.embed_dim, dtype=torch.float32))
        query *= conv_feat
        key   *= conv_feat
        scores = torch.softmax(query @ key.transpose(-2, -1) / np.sqrt(self.embed_dim), dim=-1)
        return scores.numpy()
