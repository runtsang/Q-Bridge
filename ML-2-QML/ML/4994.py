"""
Hybrid classical regression module that mirrors the EstimatorQNN example.

The class implements a configurable feed‑forward network and exposes
a FastEstimator wrapper for quick batch evaluation.  It also
provides a helper to build a matching quantum circuit that can be
used by the QML side.
"""

from __future__ import annotations

import torch
from torch import nn
from typing import Iterable, Tuple, List

# The FastEstimator utilities are lightweight and can be dropped in
# without pulling in heavy Qiskit dependencies.
from.FastEstimator import FastEstimator


def _ensure_batch(values: Iterable[float]) -> torch.Tensor:
    """Guarantee a 2‑D tensor for batch evaluation."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


def build_feedforward(
    input_dim: int, hidden_layers: int = 2, hidden_size: int = 8
) -> nn.Module:
    """Construct a simple fully‑connected regression network."""
    layers: List[nn.Module] = []
    in_dim = input_dim
    for _ in range(hidden_layers):
        layers.append(nn.Linear(in_dim, hidden_size))
        layers.append(nn.Tanh())
        in_dim = hidden_size
    layers.append(nn.Linear(hidden_size, 1))
    return nn.Sequential(*layers)


class EstimatorQNN(nn.Module):
    """
    Classical estimator that mirrors the quantum EstimatorQNN.

    Parameters
    ----------
    input_dim : int
        Number of input features.
    hidden_layers : int, default=2
        Depth of the feed‑forward network.
    hidden_size : int, default=8
        Size of each hidden layer.
    """

    def __init__(self, input_dim: int, hidden_layers: int = 2, hidden_size: int = 8):
        super().__init__()
        self.net = build_feedforward(input_dim, hidden_layers, hidden_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(inputs)

    # ------------------------------------------------------------------
    # Batch evaluation utilities
    # ------------------------------------------------------------------
    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]] | None = None,
        parameter_sets: Iterable[Iterable[float]] | None = None,
    ) -> List[List[float]]:
        """Convenience wrapper that delegates to FastEstimator."""
        estimator = FastEstimator(self)
        return estimator.evaluate(
            observables=observables or [lambda out: out.mean(dim=-1)],
            parameter_sets=parameter_sets or [],
        )

    # ------------------------------------------------------------------
    # Quantum circuit construction helper
    # ------------------------------------------------------------------
    def build_quantum_circuit(
        self, depth: int = 2
    ) -> Tuple["QuantumCircuit", Iterable["Parameter"], Iterable["Parameter"], List["SparsePauliOp"]]:
        """
        Build a Qiskit circuit that mirrors the architecture of the
        classical network.  The circuit uses `build_classifier_circuit`
        from the quantum helper module.
        """
        from.QuantumClassifierModel import build_classifier_circuit  # type: ignore
        from qiskit.circuit import ParameterVector
        from qiskit.quantum_info import SparsePauliOp

        # Align number of qubits with the network input dimension
        circuit, enc_params, var_params, observables = build_classifier_circuit(
            num_qubits=self.net[0].in_features, depth=depth
        )
        return circuit, enc_params, var_params, observables
