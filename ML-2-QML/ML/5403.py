import numpy as np
import torch
from torch import nn
from typing import Iterable, Optional

class HybridLayer(nn.Module):
    """
    Hybrid layer that blends fully connected, self-attention, and convolutional
    feature extraction with optional quantum attention. The class can operate
    in three modes:

    - 'linear': simple linear projection + tanh (like FCL)
    - 'attention': classical multi-head attention (like SelfAttention)
    - 'conv': convolutional feature extractor followed by linear projection
      (like QuantumNAT)

    The interface mirrors the seed modules: a `run` method that accepts an
    iterable of parameters and returns a NumPy array.
    """

    def __init__(self,
                 mode: str = 'linear',
                 n_features: int = 1,
                 embed_dim: int = 4,
                 n_heads: int = 1,
                 conv_channels: Optional[int] = None,
                 **kwargs):
        super().__init__()
        self.mode = mode
        self.n_features = n_features
        self.embed_dim = embed_dim
        self.n_heads = n_heads

        if mode == 'linear':
            # Simple linear layer + tanh
            self.linear = nn.Linear(n_features, 1)
        elif mode == 'attention':
            # Classical multi-head attention
            self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
            self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
            self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        elif mode == 'conv':
            if conv_channels is None:
                conv_channels = 8
            # Simple CNN followed by linear
            self.features = nn.Sequential(
                nn.Conv2d(1, conv_channels, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(conv_channels, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
            self.fc = nn.Sequential(
                nn.Linear(16 * 7 * 7, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
        else:
            raise ValueError(f"Unsupported mode {mode}")

        # Common post-processing
        self.final = nn.Tanh()

    def forward(self, x: torch.Tensor, thetas: Iterable[float] = None) -> torch.Tensor:
        if self.mode == 'linear':
            out = self.linear(x)
        elif self.mode == 'attention':
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
            scores = torch.softmax(q @ k.transpose(-2, -1) / np.sqrt(self.embed_dim), dim=-1)
            attn_out = scores @ v
            out = attn_out.mean(dim=1).unsqueeze(-1)
        elif self.mode == 'conv':
            features = self.features(x)
            flattened = features.view(features.shape[0], -1)
            out = self.fc(flattened)
        else:
            raise RuntimeError("unreachable")

        return self.final(out)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Run the layer with the provided parameters. For modes that ignore thes
        (e.g. linear, attention, conv) thetas are simply ignored.
        """
        # Use a dummy input of zeros to trigger the forward pass.
        if self.mode == 'conv':
            dummy = torch.zeros(1, 1, 28, 28)
        else:
            dummy = torch.zeros(1, self.n_features)
        out = self.forward(dummy, thetas)
        return out.detach().cpu().numpy()

__all__ = ["HybridLayer"]
