"""Hybrid estimator module: Classical GNN + Quantum backend.

This module defines UnifiedGraphEstimator that first runs a classical
graph‑neural‑network to produce a vector of parameters for a
parametrised quantum circuit, then evaluates the circuit on a quantum
backend.  It supports optional shot‑noise simulation and is fully
compatible with the interface used in the seed project.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List

import numpy as np
import torch
from torch import nn
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator

ScalarObservable = torch.Tensor | float  # placeholder for type hints


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class ClassicalGNN(nn.Module):
    """Feed‑forward GNN that emulates the architecture of the seed GraphQNN."""

    def __init__(self, arch: Sequence[int]) -> None:
        super().__init__()
        self.arch = arch
        self.layers = nn.ModuleList(
            [nn.Linear(in_f, out_f) for in_f, out_f in zip(arch[:-1], arch[1:])]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.layers:
            out = torch.tanh(layer(out))
        return out


class QuantumEstimator:
    """Evaluate expectation values of a parametrised quantum circuit."""

    def __init__(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit
        self._params = list(circuit.parameters)

    def _bind(self, values: Sequence[float]) -> QuantumCircuit:
        if len(values)!= len(self._params):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._params, values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results


class ShotNoiseEstimator(QuantumEstimator):
    """Add Gaussian shot‑noise to a deterministic quantum estimator."""

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in raw:
            noisy_row = [
                complex(
                    rng.normal(mean.real, max(1e-6, 1 / shots)),
                    rng.normal(mean.imag, max(1e-6, 1 / shots)),
                )
                for mean in row
            ]
            noisy.append(noisy_row)
        return noisy


class UnifiedGraphEstimator:
    """Hybrid estimator that generates quantum circuit parameters from a
    classical GNN and evaluates them on a quantum backend.
    """

    def __init__(
        self,
        gnn_arch: Sequence[int],
        circuit: QuantumCircuit,
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> None:
        self.gnn = ClassicalGNN(gnn_arch)
        self.quantum = ShotNoiseEstimator(circuit) if shots is not None else QuantumEstimator(circuit)
        self.shots = shots
        self.seed = seed

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Generate circuit parameters via the GNN and evaluate the circuit.

        Parameters
        ----------
        observables : iterable of BaseOperator
            Observables whose expectation values are to be computed.
        parameter_sets : sequence of sequences
            Each inner sequence is an input feature vector that will be fed
            into the GNN.

        Returns
        -------
        List[List[complex]]
            A list of rows, one per input, each containing the expectation
            values for all observables.
        """
        # First compute GNN outputs to obtain parameter vectors
        self.gnn.eval()
        with torch.no_grad():
            params = [_ensure_batch(p) for p in parameter_sets]
            param_vectors = [self.gnn(p).flatten().cpu().numpy() for p in params]

        # Evaluate the quantum circuit with the generated parameters
        return self.quantum.evaluate(observables, param_vectors)


__all__ = ["UnifiedGraphEstimator"]
