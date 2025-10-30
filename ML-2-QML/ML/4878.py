"""Quantitative hybrid classifier – classical implementation.

This module defines a `QuantumHybridClassifier` that mirrors the interface of
the original `QuantumClassifierModel` but relies solely on PyTorch and NumPy.
It supports:
* depth‑controlled dense encoding
* optional self‑attention (classical version)
* a final linear head producing binary logits

The class can be used as a drop‑in replacement for the quantum version
when a quantum backend is unavailable.  The `use_quantum` flag simply
activates a warning; the core logic remains fully classical.

Typical usage::

    from QuantumHybridClassifier import QuantumHybridClassifier
    model = QuantumHybridClassifier(num_features=64, depth=3, 
                                    attention_depth=2, use_quantum=False)
    logits = model(torch.randn(8, 64))
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["QuantumHybridClassifier"]


class _DenseEncoder(nn.Module):
    """Depth‑controlled dense encoder.

    Parameters
    ----------
    num_features : int
        Number of input features.
    depth : int
        Number of hidden layers.  Each hidden layer maps
        ``num_features → num_features``.
    """
    def __init__(self, num_features: int, depth: int) -> None:
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.append(nn.Linear(num_features, num_features))
            layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class _SelfAttention(nn.Module):
    """Classical self‑attention block.

    Implements a simple scaled dot‑product attention with trainable
    projection matrices.  The implementation follows the style of
    the reference `SelfAttention.py` seed.
    """
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        scores = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.embed_dim), dim=-1)
        return torch.matmul(scores, v)


class QuantumHybridClassifier(nn.Module):
    """Hybrid classifier that can operate in pure‑classical mode.

    Parameters
    ----------
    num_features : int
        Dimensionality of input vectors.
    depth : int
        Depth of the dense encoder.
    attention_depth : int
        Number of attention layers to stack.
    use_quantum : bool, default=False
        If True, the constructor will issue a warning that the
        quantum backend is not available.  The forward pass remains
        classical.
    """
    def __init__(self,
                 num_features: int,
                 depth: int = 2,
                 attention_depth: int = 1,
                 use_quantum: bool = False) -> None:
        super().__init__()
        self.use_quantum = use_quantum
        if self.use_quantum:
            # In the classical module we do not import qiskit.
            # The flag is only for API compatibility.
            print("[QuantumHybridClassifier] Quantum mode requested, "
                  "but this is the classical implementation. "
                  "Proceeding with classical forward.")

        self.encoder = _DenseEncoder(num_features, depth)
        self.attention_layers = nn.ModuleList(
            [_SelfAttention(num_features) for _ in range(attention_depth)]
        )
        self.classifier = nn.Linear(num_features, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, num_features)``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(batch, 2)``.
        """
        x = self.encoder(x)
        for attn in self.attention_layers:
            x = attn(x)
        logits = self.classifier(x)
        return logits
