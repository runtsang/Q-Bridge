"""Hybrid classical self‑attention with a QCNN backend.

Provides a PyTorch module that first performs a self‑attention
transform and then feeds the representation to a lightweight
fully‑connected network that emulates a quantum convolutional
neural network.  The class can be wrapped by the FastEstimator
utility to evaluate batches of parameters with optional shot
noise, mirroring the quantum estimator API.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

# --- Classical Self‑Attention --------------------------------------------
class ClassicalSelfAttention(nn.Module):
    """Pure‑Python self‑attention block used as a drop‑in replacement for the
    quantum self‑attention template.  It accepts explicit rotation and
    entanglement parameters and operates on a batch of input vectors.
    """
    def __init__(self, embed_dim: int = 4) -> None:
        super().__init__()
        self.embed_dim = embed_dim

    def forward(
        self,
        inputs: torch.Tensor,
        rotation_params: torch.Tensor,
        entangle_params: torch.Tensor,
    ) -> torch.Tensor:
        q = torch.matmul(inputs, rotation_params.reshape(self.embed_dim, -1))
        k = torch.matmul(inputs, entangle_params.reshape(self.embed_dim, -1))
        v = inputs
        scores = torch.softmax(q @ k.transpose(-2, -1) / np.sqrt(self.embed_dim), dim=-1)
        return scores @ v

# --- QCNN emulation -------------------------------------------------------
class QCNNModel(nn.Module):
    """Stack of fully‑connected layers that mimic the behaviour of a
    quantum convolutional neural network.  The architecture is
    identical to the reference QCNNModel but expressed in PyTorch.
    """
    def __init__(self) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

# --- Hybrid Module --------------------------------------------------------
class HybridSelfAttention(nn.Module):
    """Combines classical self‑attention with a QCNN block.

    The forward method accepts a batch of input vectors together
    with rotation and entanglement parameters for the attention
    stage.  The attention output is then forwarded to the QCNN
    model.  This design mirrors the quantum architecture where
    the self‑attention block is a variational circuit followed by
    a convolution‑pool‑convolution stack.
    """
    def __init__(self, embed_dim: int = 4, qcnn: nn.Module | None = None) -> None:
        super().__init__()
        self.attention = ClassicalSelfAttention(embed_dim=embed_dim)
        self.qcnn = qcnn if qcnn is not None else QCNNModel()

    def forward(
        self,
        inputs: torch.Tensor,
        rotation_params: torch.Tensor,
        entangle_params: torch.Tensor,
    ) -> torch.Tensor:
        attn_out = self.attention(inputs, rotation_params, entangle_params)
        if attn_out.shape[-1]!= 8:
            pad = nn.Linear(attn_out.shape[-1], 8, bias=False).to(attn_out.device)
            attn_out = pad(attn_out)
        return self.qcnn(attn_out)

# --- Estimator Wrapper ----------------------------------------------------
from collections.abc import Iterable, Sequence
from typing import Callable, List

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

class FastHybridEstimator:
    """Evaluates a HybridSelfAttention model over many parameter sets.

    Parameters are supplied as a list of floats.  The estimator is
    compatible with the FastBaseEstimator API: it returns a list of
    real‑valued observables for each parameter set.
    """
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                rotation = torch.as_tensor(params[:3], dtype=torch.float32)
                entangle = torch.as_tensor(params[3:], dtype=torch.float32)
                inputs = torch.eye(8, dtype=torch.float32)
                outputs = self.model(inputs, rotation, entangle)
                row: List[float] = []
                for observable in observables:
                    val = observable(outputs)
                    if isinstance(val, torch.Tensor):
                        scalar = float(val.mean().cpu())
                    else:
                        scalar = float(val)
                    row.append(scalar)
                results.append(row)
        return results

# --- Factory ---------------------------------------------------------------
def HybridSelfAttentionFactory() -> nn.Module:
    """Convenience factory that returns a ready‑to‑use HybridSelfAttention."""
    return HybridSelfAttention()

__all__ = [
    "HybridSelfAttention",
    "HybridSelfAttentionFactory",
    "FastHybridEstimator",
    "QCNNModel",
    "ClassicalSelfAttention",
]
