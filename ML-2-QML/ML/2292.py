"""Unified classical estimator with a quantum expectation head.

The module defines `UnifiedEstimatorQNN`, a PyTorch model that
combines a scalable dense feature extractor with a hybrid layer
that forwards through an external quantum function.  The quantum
function is supplied by the `qml` module and is invoked via a
lightweight autograd wrapper that implements a parameter‑shift
finite‑difference for back‑propagation.

This design unifies the simple EstimatorQNN feed‑forward seed
with the hybrid convolutional network from ClassicalQuantumBinaryClassification,
but keeps the classical side fully PyTorch‑compatible and
quantum‑agnostic.
"""

from __future__ import annotations

import torch
from torch import nn
import numpy as np
from typing import Callable, Iterable, Tuple, Optional

# --------------------------------------------------------------------------- #
# Classical feature extractor
# --------------------------------------------------------------------------- #
class _FeatureExtractor(nn.Module):
    """Scalable dense feature extractor.

    The architecture grows with the input dimensionality, using a
    stack of linear layers with tanh activations.  The depth and width
    are chosen to provide a non‑linear expansion while keeping the
    number of parameters moderate.
    """
    def __init__(self, input_dim: int, hidden_dims: Optional[Iterable[int]] = None) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32, 16]
        layers = []
        last_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.Tanh())
            last_dim = h
        layers.append(nn.Linear(last_dim, 1))  # output dimension
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# --------------------------------------------------------------------------- #
# Autograd wrapper for an external quantum function
# --------------------------------------------------------------------------- #
class QuantumAutogradFunction(torch.autograd.Function):
    """Autograd wrapper that forwards to an external quantum function.

    Parameters
    ----------
    quantum_fn : Callable[[np.ndarray], np.ndarray]
        A callable that accepts a 1‑D numpy array of parameters and
        returns a 1‑D numpy array of expectation values.  The function
        must be differentiable via a parameter‑shift rule implemented
        in the backward method.
    shift : float
        The shift value used for the parameter‑shift finite‑difference.
    """
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, quantum_fn: Callable[[np.ndarray], np.ndarray], shift: float) -> torch.Tensor:
        ctx.quantum_fn = quantum_fn
        ctx.shift = shift
        ctx.save_for_backward(inputs)
        inputs_np = inputs.detach().cpu().numpy()
        out_np = ctx.quantum_fn(inputs_np)
        return torch.tensor(out_np, dtype=inputs.dtype, device=inputs.device)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None]:
        inputs, = ctx.saved_tensors
        shift = ctx.shift
        quantum_fn = ctx.quantum_fn
        grad_inputs = []
        for i in range(inputs.shape[0]):
            plus = inputs.clone()
            minus = inputs.clone()
            plus[i] += shift
            minus[i] -= shift
            f_plus = quantum_fn(plus.detach().cpu().numpy())
            f_minus = quantum_fn(minus.detach().cpu().numpy())
            grad = (f_plus - f_minus) / (2 * shift)
            grad_inputs.append(grad)
        grad_inputs = torch.tensor(grad_inputs, dtype=inputs.dtype, device=inputs.device)
        return grad_inputs * grad_output, None, None

# --------------------------------------------------------------------------- #
# Hybrid layer that forwards through a quantum expectation function
# --------------------------------------------------------------------------- #
class HybridLayer(nn.Module):
    """Hybrid layer that forwards through a quantum expectation function."""
    def __init__(self, quantum_fn: Callable[[np.ndarray], np.ndarray], shift: float = np.pi / 2) -> None:
        super().__init__()
        self.quantum_fn = quantum_fn
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten to 1‑D as expected by the quantum function
        flat = x.view(-1)
        return QuantumAutogradFunction.apply(flat, self.quantum_fn, self.shift)

# --------------------------------------------------------------------------- #
# Unified hybrid estimator
# --------------------------------------------------------------------------- #
class UnifiedEstimatorQNN(nn.Module):
    """Full hybrid estimator combining a classical feature extractor
    with a quantum expectation head.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input feature vector.
    hidden_dims : Optional[Iterable[int]]
        Hidden layer sizes for the feature extractor.
    quantum_fn : Optional[Callable[[np.ndarray], np.ndarray]]
        Quantum function that maps a 1‑D parameter vector to a
        1‑D expectation value.  If ``None`` a simple linear head is
        used.
    """
    def __init__(self,
                 input_dim: int,
                 hidden_dims: Optional[Iterable[int]] = None,
                 quantum_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                 shift: float = np.pi / 2) -> None:
        super().__init__()
        self.feature_extractor = _FeatureExtractor(input_dim, hidden_dims)
        if quantum_fn is not None:
            self.hybrid = HybridLayer(quantum_fn, shift)
        else:
            # Fallback linear head
            self.hybrid = nn.Linear(self.feature_extractor.net[-1].out_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        out = self.hybrid(features)
        return out

__all__ = ["UnifiedEstimatorQNN", "_FeatureExtractor", "HybridLayer", "QuantumAutogradFunction"]
