"""QuantumClassifierModel implementation for a variational circuit.

The class builds a parameter‑shiftable ansatz, simulates the circuit with
Qiskit Aer, and provides a simple training routine that optimises
the expectation value of a set of Pauli‑Z observables.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
from qiskit import QuantumCircuit, Aer
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.opflow import StateFn, PauliExpectation, CircuitSampler

class QuantumClassifierModel:
    """Variational classifier ansatz with data‑encoding and a trainable depth.

    Parameters
    ----------
    num_qubits : int
        Number of qubits (features).
    depth : int
        Number of variational layers.
    backend : str | qiskit.providers.Backend, default="statevector_simulator"
        Backend used for simulation.
    """

    def __init__(self, num_qubits: int, depth: int, backend: str | None = "statevector_simulator"):
        self.num_qubits = num_qubits
        self.depth = depth
        self.backend = Aer.get_backend(backend) if isinstance(backend, str) else backend
        self.circuit, self.encoding, self.weights, self.observables = self._build_circuit()

    def _build_circuit(self) -> Tuple[QuantumCircuit, List[ParameterVector], List[ParameterVector], List[SparsePauliOp]]:
        """Create a layered ansatz with RX data encoding and Ry–CZ rotations."""
        encoding = ParameterVector("x", self.num_qubits)
        weights = ParameterVector("theta", self.num_qubits * self.depth)

        qc = QuantumCircuit(self.num_qubits)
        for qubit, param in enumerate(encoding):
            qc.rx(param, qubit)

        idx = 0
        for _ in range(self.depth):
            for qubit in range(self.num_qubits):
                qc.ry(weights[idx], qubit)
                idx += 1
            for qubit in range(self.num_qubits - 1):
                qc.cz(qubit, qubit + 1)

        observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (self.num_qubits - i - 1))
            for i in range(self.num_qubits)
        ]
        return qc, list(encoding), list(weights), observables

    def encode(self, x: np.ndarray) -> List[float]:
        """Return a list of parameter values for the encoding."""
        return x.tolist()

    def circuit_with_params(self, params: List[float]) -> QuantumCircuit:
        """Instantiate a circuit with concrete parameter values."""
        param_values = {p: v for p, v in zip(self.encoding + self.weights, params)}
        return self.circuit.bind_parameters(param_values)

    def expectation(self, params: List[float]) -> float:
        """Compute expectation value of the sum of Z observables."""
        qc = self.circuit_with_params(params)
        total = 0.0
        for op in self.observables:
            sampler = CircuitSampler(self.backend).convert(StateFn(op, qc) @ PauliExpectation())
            total += sampler.eval().real
        return total

    def train(self, x_train: np.ndarray, y_train: np.ndarray, epochs: int = 20, lr: float = 0.01) -> List[float]:
        """Gradient‑free optimisation using the parameter‑shift rule."""
        # initialise parameters randomly
        params = np.random.randn(len(self.encoding) + len(self.weights))
        loss_history = []

        for epoch in range(epochs):
            for x, y in zip(x_train, y_train):
                exp_val = self.expectation(params)
                pred = 1 if exp_val > 0 else 0
                loss = - (y * np.log(pred + 1e-6) + (1 - y) * np.log(1 - pred + 1e-6))

                grads = np.zeros_like(params)
                shift = np.pi / 2
                for i in range(len(params)):
                    shift_vec = np.zeros_like(params)
                    shift_vec[i] = shift
                    f_plus = self.expectation(params + shift_vec)
                    f_minus = self.expectation(params - shift_vec)
                    grads[i] = (f_plus - f_minus) / (2 * shift)

                params -= lr * grads
            loss_history.append(loss)
        return loss_history

    def predict(self, x: np.ndarray, threshold: float = 0.0) -> np.ndarray:
        """Return binary predictions for a batch of samples."""
        preds = []
        for sample in x:
            exp_val = self.expectation(np.concatenate((self.encode(sample), np.zeros(len(self.weights)))))
            preds.append(1 if exp_val > threshold else 0)
        return np.array(preds)

    def get_parameters(self) -> List[float]:
        """Return current trainable parameters."""
        return np.zeros(len(self.encoding) + len(self.weights))
