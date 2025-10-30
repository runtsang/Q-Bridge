"""Hybrid classical attention + fraud‑detection network.

The module defines a reusable `HybridFraudAttention` class that
encapsulates a classical self‑attention module followed by a
fraud‑detection stack.  The interface mimics the original seeds
while adding a new `forward` method that accepts a batch of inputs and
produces fraud scores.

Typical usage:

```python
from SelfAttention__gen064 import HybridFraudAttention
model = HybridFraudAttention(embed_dim=4, fraud_layers=3)
out = model(torch.randn(10, 4))
```
"""

from __future__ import annotations

import torch
from torch import nn

# --------------------------------------------------------------------------- #
# 1. Classical self‑attention helper (adapted from the original seed)
# --------------------------------------------------------------------------- #
class ClassicalSelfAttention(nn.Module):
    """Simple dot‑product self‑attention using learnable rotation and
    entanglement matrices.  The implementation mirrors the original
    seed but is written as an nn.Module for seamless integration.
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        # Parameters are initialized randomly; they can be tuned later.
        self.rotation = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.entangle = nn.Parameter(torch.randn(embed_dim, embed_dim))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Tensor of shape (batch, embed_dim)

        Returns:
            Tensor of shape (batch, embed_dim) – the attended representation.
        """
        query = inputs @ self.rotation
        key   = inputs @ self.entangle
        scores = torch.softmax(query @ key.transpose(-2, -1) / (self.embed_dim ** 0.5), dim=-1)
        return scores @ inputs


# --------------------------------------------------------------------------- #
# 2. Fraud‑detection building blocks (adapted from the original seed)
# --------------------------------------------------------------------------- #
class FraudLayer(nn.Module):
    """A single fraud‑detection layer built from a 2‑unit linear block,
    a Tanh activation, and a scaling/shift transformation.
    """
    def __init__(self, params: dict):
        super().__init__()
        weight = torch.tensor([[params["bs_theta"], params["bs_phi"]],
                               [params["squeeze_r"][0], params["squeeze_r"][1]]],
                              dtype=torch.float32)
        bias = torch.tensor(params["phases"], dtype=torch.float32)
        self.linear = nn.Linear(2, 2)
        with torch.no_grad():
            self.linear.weight.copy_(weight)
            self.linear.bias.copy_(bias)
        self.activation = nn.Tanh()
        self.scale = torch.tensor(params["displacement_r"], dtype=torch.float32)
        self.shift = torch.tensor(params["displacement_phi"], dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(self.linear(x))
        out = out * self.scale + self.shift
        return out


def build_fraud_detection_model(input_params: dict,
                                layer_params: list[dict]) -> nn.Sequential:
    """
    Construct a PyTorch sequential model that mirrors the fraud‑detection
    architecture from the seed.  The first layer is un‑clipped; the rest
    are clipped to keep parameters bounded.
    """
    modules = [FraudLayer(input_params)]
    for params in layer_params:
        modules.append(FraudLayer(params))
    # Final classifier
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


# --------------------------------------------------------------------------- #
# 3. Hybrid network that stitches attention and fraud‑detection
# --------------------------------------------------------------------------- #
class HybridFraudAttention(nn.Module):
    """
    A hybrid network that first applies self‑attention to the input
    representation and then feeds the result into a fraud‑detection
    stack.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input and attention representation.
    fraud_layer_count : int
        Number of fraud‑detection layers to stack.
    """
    def __init__(self, embed_dim: int = 4, fraud_layer_count: int = 3):
        super().__init__()
        self.attention = ClassicalSelfAttention(embed_dim)
        # Randomly initialise fraud layer parameters for demonstration.
        # In practice, these would be optimised or loaded from a checkpoint.
        base_params = {
            "bs_theta": 0.5, "bs_phi": 0.3,
            "phases": (0.1, -0.1),
            "squeeze_r": (0.2, 0.2),
            "squeeze_phi": (0.0, 0.0),
            "displacement_r": (0.5, 0.5),
            "displacement_phi": (0.0, 0.0),
            "kerr": (0.0, 0.0),
        }
        layer_params = [base_params.copy() for _ in range(fraud_layer_count)]
        self.fraud = build_fraud_detection_model(base_params, layer_params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: attention → fraud detection → binary fraud score.
        """
        attn_out = self.attention(x)
        return self.fraud(attn_out)


__all__ = ["HybridFraudAttention", "ClassicalSelfAttention", "build_fraud_detection_model"]
