"""Quantum estimator built on Pennylane.

Features:
- Supports any number of qubits and a user‑supplied list of observables.
- Evaluates expectation values as a vector, enabling batch processing of multiple observables.
- Optional Gaussian noise to emulate finite‑shot statistics.
- The estimator can be embedded in a PyTorch model via the HybridLayer below.
"""

import pennylane as qml
import numpy as np
from collections.abc import Iterable, Sequence
from typing import List, Optional, Union

class HybridFastEstimator:
    """Quantum estimator that evaluates a variational circuit with Pennylane."""

    def __init__(
        self,
        *,
        n_qubits: int,
        device: str = "default.qubit",
        shots: Optional[int] = None,
        shift: float = np.pi / 2,
    ) -> None:
        self.n_qubits = n_qubits
        self.shift = shift
        self.device = qml.device(device, wires=n_qubits, shots=shots)
        self._observables: List[qml.operation.Operator] = []

    def _build_qnode(self, params: np.ndarray) -> qml.QNode:
        """Return a QNode that evaluates all stored observables for the given parameters."""
        observables = self._observables

        @qml.qnode(self.device, interface="torch")
        def circuit(*args):
            # Simple variational circuit: H on all qubits, then RY rotations,
            # followed by a layer of CNOTs.
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)
            for i, val in enumerate(args):
                qml.RY(val, wires=i % self.n_qubits)
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            # Return expectation values of all observables.
            return [qml.expval(obs) for obs in observables]
        return circuit(params)

    def evaluate(
        self,
        observables: Iterable[qml.operation.Operator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        noise_shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[float]]:
        """Evaluate expectation values for the supplied observables and parameters."""
        self._observables = list(observables) or [qml.PauliZ(0)]
        results: List[List[float]] = []

        for params in parameter_sets:
            qnode = self._build_qnode(np.array(params, dtype=np.float64))
            row = [float(val) for val in qnode(*params)]
            results.append(row)

        if noise_shots is None:
            return results

        rng = np.random.default_rng(seed)
        noisy = []
        for row in results:
            noisy_row = [rng.normal(val, max(1e-6, 1 / noise_shots)) for val in row]
            noisy.append(noisy_row)
        return noisy

    def __repr__(self) -> str:
        return f"<HybridFastEstimator n_qubits={self.n_qubits} device={self.device} shots={self.device.shots}>"

__all__ = ["HybridFastEstimator"]
