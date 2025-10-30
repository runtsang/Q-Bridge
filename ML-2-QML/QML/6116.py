"""Quantum classifier circuit factory with parameter‑shift training support."""

from __future__ import annotations

from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.providers.aer import AerSimulator
import numpy as np


def build_classifier_circuit(
    num_qubits: int,
    depth: int,
    entanglement: str = "full",
    basis: str = "rx",
) -> Tuple[QuantumCircuit, List[ParameterVector], List[ParameterVector], List[SparsePauliOp]]:
    """
    Construct a layered ansatz with explicit data encoding and variational parameters.
    Parameters:
        num_qubits: number of qubits (features).
        depth: number of variational layers.
        entanglement: 'full', 'linear', or 'none' to control CX pattern.
        basis: 'rx' or 'ry' for the single‑qubit rotation used in the encoding.
    Returns:
        circuit, encoding parameters, variational parameters, observables.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)

    # Data encoding
    for param, qubit in zip(encoding, range(num_qubits)):
        if basis == "rx":
            circuit.rx(param, qubit)
        else:
            circuit.ry(param, qubit)

    idx = 0
    for _ in range(depth):
        # Variational rotations
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        # Entangling layer
        if entanglement == "full":
            for q in range(num_qubits):
                for r in range(q + 1, num_qubits):
                    circuit.cz(q, r)
        elif entanglement == "linear":
            for q in range(num_qubits - 1):
                circuit.cz(q, q + 1)
        # 'none' skips entanglement

    # Observables: Z on each qubit
    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]
    return circuit, list(encoding), list(weights), observables


class QuantumClassifierModel:
    """
    Hybrid interface that trains a variational circuit using the parameter‑shift rule.
    Mirrors the classical API: train() and evaluate() methods.
    """

    def __init__(
        self,
        num_qubits: int,
        depth: int,
        entanglement: str = "full",
        basis: str = "rx",
        shots: int = 1024,
        lr: float = 0.01,
        epochs: int = 30,
    ):
        self.circuit, self.encoding, self.weights, self.observables = build_classifier_circuit(
            num_qubits, depth, entanglement, basis
        )
        self.shots = shots
        self.lr = lr
        self.epochs = epochs

        # Compile simulator
        self.backend = AerSimulator()
        self.backend.options.shots = shots

        # Parameter mapping
        self.param_values = np.zeros(len(self.weights))

    def _expectation(self, param_values: List[float]) -> np.ndarray:
        """
        Evaluate expectation values of the observables for a given set of parameters
        using the AerSimulator.  Returns a numpy array of shape (num_qubits,).
        """
        bound_circuit = self.circuit.bind_parameters(
            dict(zip(self.weights, param_values))
        )
        job = execute(bound_circuit, self.backend)
        result = job.result()
        counts = result.get_counts()
        exp_vals = np.zeros(len(self.observables))
        for state, freq in counts.items():
            prob = freq / self.shots
            for i, _ in enumerate(self.observables):
                bit = int(state[::-1][i])  # Qiskit returns little‑endian
                exp_vals[i] += prob * (1 if bit == 0 else -1)
        return exp_vals

    def _parameter_shift_grad(self, param_values: List[float], idx: int) -> float:
        """
        Compute gradient of the loss w.r.t a single parameter using the
        parameter‑shift rule.  The loss is the negative log‑likelihood of the
        target label (assumed to be 0 or 1 for binary classification).
        """
        shift = np.pi / 2
        plus = param_values.copy()
        minus = param_values.copy()
        plus[idx] += shift
        minus[idx] -= shift
        exp_plus = self._expectation(plus)
        exp_minus = self._expectation(minus)
        # For binary classification we use the first observable as log‑likelihood
        return 0.5 * (exp_plus[0] - exp_minus[0])

    def train(self, X: np.ndarray, y: np.ndarray, verbose: bool = False) -> None:
        """
        Simple gradient‑descent loop using the parameter‑shift rule.
        X is an array of shape (n_samples, num_qubits).  y contains 0/1 labels.
        """
        for epoch in range(self.epochs):
            grads = np.zeros_like(self.param_values)
            for x, label in zip(X, y):
                # Bind data to the circuit
                bound = self.circuit.bind_parameters(dict(zip(self.encoding, x)))
                # Compute expectation for the current parameters
                exp_vals = self._expectation(self.param_values)
                # Compute loss gradient w.r.t each parameter
                for i in range(len(self.param_values)):
                    grads[i] += self._parameter_shift_grad(self.param_values, i)
            grads /= len(X)
            self.param_values -= self.lr * grads
            if verbose and epoch % 5 == 0:
                loss = -np.mean(
                    [
                        np.log(0.5 + 0.5 * self._expectation(self.param_values)[0])
                        for _ in X
                    ]
                )
                print(f"Epoch {epoch}: loss={loss:.4f}")

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return classification accuracy on the provided data.
        """
        correct = 0
        for x, label in zip(X, y):
            exp_vals = self._expectation(self.param_values)
            pred = 0 if exp_vals[0] > 0 else 1
            if pred == label:
                correct += 1
        return correct / len(y)


__all__ = ["build_classifier_circuit", "QuantumClassifierModel"]
