from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class ConvFilter(nn.Module):
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def run(self, data: Tensor) -> float:
        tensor = torch.as_tensor(data, dtype=torch.float32)
        tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean().item()

class GraphNetwork(nn.Module):
    def __init__(self, layer_sizes: list[int]):
        super().__init__()
        self.layers = nn.ModuleList()
        for in_f, out_f in zip(layer_sizes[:-1], layer_sizes[1:]):
            self.layers.append(nn.Linear(in_f, out_f))

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = torch.tanh(layer(x))
        return x

class Estimator(nn.Module):
    def __init__(self, in_features: int = 4, hidden: int = 8, out_features: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.Tanh(),
            nn.Linear(hidden, out_features),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, embed_dim),
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: Tensor) -> Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class HybridConvGraphEstimator(nn.Module):
    """
    Hybrid architecture that chains a classical convolution filter,
    a graph neural network, a transformer block, and a regression head.
    """
    def __init__(
        self,
        conv_kernel: int = 2,
        conv_threshold: float = 0.0,
        graph_arch: list[int] = [1, 2, 2],
        transformer_params: dict | None = None,
        estimator_params: dict | None = None,
    ):
        super().__init__()
        self.conv = ConvFilter(conv_kernel, conv_threshold)
        self.graph_net = GraphNetwork(graph_arch)
        tp = transformer_params or {"embed_dim": 2, "num_heads": 1, "ffn_dim": 4}
        self.transformer = TransformerBlock(**tp)
        ep = estimator_params or {"in_features": 2, "hidden": 8, "out_features": 1}
        self.estimator = Estimator(**ep)

    def run(self, data):
        """
        Accepts a 2D array of shape (kernel_size, kernel_size).
        Returns a scalar prediction.
        """
        conv_out = self.conv.run(data)  # scalar
        graph_in = torch.tensor([conv_out], dtype=torch.float32).unsqueeze(0)  # (1,1)
        graph_out = self.graph_net(graph_in)  # (1,2)
        seq = graph_out.repeat(1, 3, 1)  # (1,3,2)
        trans_out = self.transformer(seq)  # (1,3,2)
        trans_mean = trans_out.mean(dim=1)  # (1,2)
        pred = self.estimator(trans_mean)  # (1,1)
        return pred.item()

__all__ = ["HybridConvGraphEstimator"]
