"""Unified hybrid estimator for classical and quantum modules.

The module implements a FastBaseEstimator that can run either a pure
PyTorch network or a TorchQuantum circuit.  The estimator is
parameter‑agnostic and supports optional Gaussian shot noise.  The
class hierarchy mirrors the original FastBaseEstimator/FastEstimator
pairs while adding a ``HybridQuantumLayer`` that can wrap any
PyTorch module and delegate to a quantum device when a flag is set.

The design choices are a direct synthesis of the 4 reference pairs:
* From FastBaseEstimator.py we keep the deterministic evaluator and
  the shot‑noise wrapper.
* From QLSTM.py and QTransformerTorch.py we borrow the idea of
  *replaceable* sub‑modules and expose a quantum fallback.
* From QuantumKernelMethod.py we borrow the RBF kernel implementation
  and expose a quantum kernel that can be used as an observable.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import torch
from torch import nn

# --------------------------------------------------------------------------- #
# Helper: observable helpers
# --------------------------------------------------------------------------- #
ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Return a 2‑D float tensor with a leading batch dimension."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

# --------------------------------------------------------------------------- #
# Base estimator – deterministic
# --------------------------------------------------------------------------- #
class FastBaseEstimator:
    """Evaluate a PyTorch or TorchQuantum model for a batch of
    parameter sets and a collection of observables.

    Parameters
    ----------
    model : nn.Module
        The model must expose a ``forward`` that accepts a 2‑D tensor
        of shape (batch, features) and returns a tensor of shape
        (batch, out_dim).  The model may be a plain nn.Module or a
        quantum module that returns a probability distribution.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Run the model once and compute each observable."""
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.model(inputs)
                row: List[float] = []
                for observable in observables:
                    value = observable(outputs)
                    if isinstance(value, torch.Tensor):
                        scalar = float(value.mean().cpu())
                    else:
                        scalar = float(value)
                    row.append(scalar)
                results.append(row)
        return results

# --------------------------------------------------------------------------- #
# Shot‑noise wrapper
# --------------------------------------------------------------------------- #
class FastEstimator(FastBaseEstimator):
    """Add Gaussian shot‑noise to the deterministic estimator."""

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

# --------------------------------------------------------------------------- #
# Hybrid quantum layer
# --------------------------------------------------------------------------- #
class HybridQuantumLayer(nn.Module):
    """
    A wrapper that can run a classical sub‑module or a quantum module
    depending on the *use_quantum* flag.  The quantum module must be a
    subclass of :class:`torchquantum.QuantumModule` and provide a
    ``forward`` that accepts a 2‑D tensor of shape (batch, features)
    and returns a probability distribution or a state vector.
    """

    def __init__(
        self,
        classical: nn.Module,
        quantum: nn.Module,
        *,
        use_quantum: bool = False,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.classical = classical
        self.quantum = quantum
        self.use_quantum = use_quantum
        self.device = device or torch.device("cpu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_quantum:
            # Ensure the quantum device is on the right device
            if hasattr(self.quantum, "q_device"):
                self.quantum.q_device = self.quantum.q_device.to(self.device)
            return self.quantum(x.to(self.device))
        else:
            return self.classical(x)

# --------------------------------------------------------------------------- #
# Example quantum kernel observable
# --------------------------------------------------------------------------- #
class QuantumKernelObservable:
    """
    Wraps a TorchQuantum kernel as a callable observable.
    The kernel returns a complex amplitude; we take its magnitude
    as a real observable.
    """

    def __init__(self, kernel: nn.Module):
        self.kernel = kernel

    def __call__(self, outputs: torch.Tensor) -> torch.Tensor:
        # Assume outputs are 2‑D (batch, out_dim) and we treat each
        # row as a quantum state; we take the kernel between the
        # state and itself.
        return torch.abs(self.kernel(outputs, outputs).diag())

# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
__all__ = [
    "FastBaseEstimator",
    "FastEstimator",
    "HybridQuantumLayer",
    "QuantumKernelObservable",
]
