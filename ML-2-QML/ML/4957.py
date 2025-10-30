from __future__ import annotations

from typing import Iterable, List
import numpy as np
import torch
from torch import nn

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

class SimpleAttention(nn.Module):
    """Light‑weight dot‑product attention used when ``use_attention`` is True."""
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, embed_dim]
        q = x
        k = x
        v = x
        scores = torch.softmax(q @ k.transpose(-2, -1) / np.sqrt(self.embed_dim), dim=-1)
        return scores @ v

class HybridFullyConnectedLayer(nn.Module):
    """
    Classical hybrid layer that unifies the linear fully‑connected block from FCL,
    optional self‑attention from SelfAttention, and fraud‑detection style weight
    clipping from FraudDetection. The layer can be expanded with multiple depths.
    """
    def __init__(
        self,
        n_features: int = 1,
        depth: int = 1,
        use_attention: bool = False,
        use_fraud_clip: bool = False,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.depth = depth
        self.use_attention = use_attention
        self.use_fraud_clip = use_fraud_clip

        modules: List[nn.Module] = []
        in_dim = n_features
        for _ in range(depth):
            linear = nn.Linear(in_dim, n_features)
            if use_fraud_clip:
                with torch.no_grad():
                    linear.weight.data = torch.clamp(linear.weight.data, -5.0, 5.0)
                    linear.bias.data   = torch.clamp(linear.bias.data,   -5.0, 5.0)
            modules.append(linear)
            modules.append(nn.Tanh())
            modules.append(nn.ReLU())
            in_dim = n_features

        if use_attention:
            self.attention = SimpleAttention(n_features)
            modules.append(self.attention)

        head = nn.Linear(n_features, 1)
        modules.append(head)

        self.network = nn.Sequential(*modules)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.network(inputs.unsqueeze(0)).squeeze(0)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Accepts a list of parameters, feeds them through the network and
        returns a single expectation value.  Mimics the original FCL interface.
        """
        values = torch.as_tensor(list(thetas), dtype=torch.float32)
        expectation = self.network(values.unsqueeze(0)).mean()
        return expectation.detach().numpy()

def FCL() -> nn.Module:
    """
    Factory compatible with the original FCL anchor.
    Returns a hybrid fully‑connected layer with depth 2, attention and fraud clipping enabled.
    """
    return HybridFullyConnectedLayer(
        n_features=1,
        depth=2,
        use_attention=True,
        use_fraud_clip=True,
    )
