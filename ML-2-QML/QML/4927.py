"""Quantum regression estimator built with Qiskit primitives and a simple linear post‑processing head.

It mirrors the classical EstimatorQNN but replaces the feed‑forward network with a variational
ansatz and a state‑vector simulator.  The class also offers a FastEstimator‑style API for
batch evaluation and optional shot noise, making it a drop‑in replacement for the
classical version.
"""

from qiskit.circuit import ParameterVector
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.primitives import StatevectorEstimator
from collections.abc import Iterable, Sequence
from typing import Optional, List
import torch
import torch.nn as nn
import numpy as np


class FastBaseEstimator:
    """Evaluates a parametrized circuit on a collection of parameter sets."""
    def __init__(self, circuit: QuantumCircuit):
        self.circuit = circuit
        self.params = list(circuit.parameters)

    def _bind(self, values: Sequence[float]) -> QuantumCircuit:
        if len(values)!= len(self.params):
            raise ValueError("Parameter count mismatch.")
        mapping = dict(zip(self.params, values))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[SparsePauliOp],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        if shots is None:
            return results
        rng = np.random.default_rng(seed)
        noisy = []
        for row in results:
            noisy_row = [
                complex(
                    rng.normal(val.real, 1 / shots) + 1j * rng.normal(val.imag, 1 / shots)
                )
                for val in row
            ]
            noisy.append(noisy_row)
        return noisy


def _build_regression_circuit(num_qubits: int, depth: int) -> QuantumCircuit:
    """Encoder + variational ansatz."""
    enc = ParameterVector("x", num_qubits)
    var = ParameterVector("theta", num_qubits * depth)
    qc = QuantumCircuit(num_qubits)
    for p, q in zip(enc, range(num_qubits)):
        qc.ry(p, q)
    idx = 0
    for _ in range(depth):
        for q in range(num_qubits):
            qc.rx(var[idx], q)
            idx += 1
        for q in range(num_qubits - 1):
            qc.cz(q, q + 1)
    return qc


def _build_observables(num_qubits: int) -> list[SparsePauliOp]:
    return [
        SparsePauliOp.from_list([("Z" * i + "I" * (num_qubits - i), 1)])
        for i in range(num_qubits)
    ]


class EstimatorQNN:
    """Quantum regression estimator that can be called like the classical counterpart."""
    def __init__(self, num_qubits: int = 2, depth: int = 2, head_weights: Sequence[float] | None = None):
        self.circuit = _build_regression_circuit(num_qubits, depth)
        self.observables = _build_observables(num_qubits)
        self.estimator = StatevectorEstimator()
        # simple linear head
        self.head_weights = torch.tensor(
            head_weights if head_weights is not None else [1.0] * num_qubits, dtype=torch.float32
        )
        self.head_bias = torch.tensor(0.0, dtype=torch.float32)

    def forward(self, params: Sequence[float]) -> float:
        """Evaluate a single parameter set."""
        bound = self.circuit.assign_parameters(dict(zip(self.circuit.parameters, params)), inplace=False)
        state = Statevector.from_instruction(bound)
        features = torch.tensor(
            [state.expectation_value(obs).real for obs in self.observables], dtype=torch.float32
        )
        return float((features @ self.head_weights + self.head_bias).item())

    def evaluate(
        self,
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[complex]]:
        estimator = FastBaseEstimator(self.circuit)
        return estimator.evaluate(self.observables, parameter_sets, shots=shots, seed=seed)


__all__ = ["EstimatorQNN", "FastBaseEstimator"]
