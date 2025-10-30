"""Quantum classifier using Qiskit with parameter‑shift gradient support."""

from __future__ import annotations

from typing import Iterable, Tuple, List

import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector


def build_classifier_circuit(
    num_qubits: int,
    depth: int = 2,
    entangler_type: str = "CZ",
    rotate_axes: Tuple[str, str] = ("X", "Y"),
) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """
    Build a layered parameter‑shift ansatz with optional entanglement pattern.

    Parameters
    ----------
    num_qubits : int
        Number of qubits / features.
    depth : int, default 2
        Number of variational layers.
    entangler_type : str, default "CZ"
        Choice of two‑qubit gate for the entangling layer.
    rotate_axes : tuple[str, str], default ("X", "Y")
        Axes used for the single‑qubit rotations in the variational layer.

    Returns
    -------
    circuit : QuantumCircuit
        The constructed circuit.
    encoding : list[Parameter]
        Data‑encoding parameters.
    weights : list[Parameter]
        Variational parameters.
    observables : list[SparsePauliOp]
        Z‑observables on each qubit.
    """
    encoding = ParameterVector("x", num_qubits)
    weight_sizes = num_qubits * depth
    weights = ParameterVector("theta", weight_sizes)

    qc = QuantumCircuit(num_qubits)

    # Data encoding
    for qubit, param in enumerate(encoding):
        qc.rx(param, qubit)

    idx = 0
    for _ in range(depth):
        # Rotation layer
        for qubit in range(num_qubits):
            axis = rotate_axes[qubit % len(rotate_axes)]
            if axis == "X":
                qc.rx(weights[idx], qubit)
            elif axis == "Y":
                qc.ry(weights[idx], qubit)
            else:
                qc.rz(weights[idx], qubit)
            idx += 1

        # Entangling layer
        if entangler_type == "CZ":
            for qubit in range(num_qubits - 1):
                qc.cz(qubit, qubit + 1)
        elif entangler_type == "CX":
            for qubit in range(num_qubits - 1):
                qc.cx(qubit, qubit + 1)

    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]

    return qc, list(encoding), list(weights), observables


class QuantumClassifierModel:
    """
    Quantum classifier that evaluates expectation values of Z on each qubit.

    Provides a parameter‑shift gradient routine and a softmax conversion
    to class probabilities for side‑by‑side comparison with the classical model.
    """

    def __init__(
        self,
        num_qubits: int,
        depth: int = 2,
        *,
        entangler_type: str = "CZ",
        rotate_axes: Tuple[str, str] = ("X", "Y"),
    ):
        self.circuit, self.encoding, self.weights, self.observables = build_classifier_circuit(
            num_qubits,
            depth,
            entangler_type=entangler_type,
            rotate_axes=rotate_axes,
        )
        self.num_qubits = num_qubits
        self.backend = Aer.get_backend("statevector_simulator")

    def _expectation(self, state: Statevector, pauli: SparsePauliOp) -> float:
        """Compute expectation value of a Pauli operator on a statevector."""
        return np.real(state.expectation_value(pauli))

    def _evaluate(self, data: np.ndarray) -> np.ndarray:
        """
        Compute expectation values for a single data sample.

        Parameters
        ----------
        data : np.ndarray, shape (num_qubits,)
            Classical input vector to be encoded into the circuit.

        Returns
        -------
        np.ndarray, shape (num_qubits,)
            Raw logits (expectation values of Z on each qubit).
        """
        bound = {p: v for p, v in zip(self.encoding, data)}
        job = execute(self.circuit.bind_parameters(bound), self.backend, shots=1)
        state = Statevector(job.result().get_statevector(self.circuit.bind_parameters(bound)))
        return np.array([self._expectation(state, obs) for obs in self.observables])

    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Forward pass returning logits (raw expectation values).
        """
        return self._evaluate(data)

    def predict_proba(self, data: np.ndarray) -> np.ndarray:
        """
        Convert raw logits to probabilities using softmax across the two classes.
        """
        logits = self.predict(data)
        exp = np.exp(logits)
        return exp / exp.sum()

    def _evaluate_shift(self, data: np.ndarray, shift_vector: List[float]) -> np.ndarray:
        """Helper: evaluate with a shifted weight vector."""
        bound = {p: v for p, v in zip(self.encoding, data)}
        for p, s in zip(self.weights, shift_vector):
            bound[p] = s
        job = execute(self.circuit.bind_parameters(bound), self.backend, shots=1)
        state = Statevector(job.result().get_statevector(self.circuit.bind_parameters(bound)))
        return np.array([self._expectation(state, obs) for obs in self.observables])

    def gradient(self, data: np.ndarray) -> np.ndarray:
        """
        Compute gradients w.r.t. variational parameters using the parameter‑shift rule.

        Parameters
        ----------
        data : np.ndarray, shape (num_qubits,)
            Input vector for which the gradient is computed.

        Returns
        -------
        np.ndarray, shape (num_params,)
            Gradient of the summed logits with respect to each variational parameter.
        """
        shift = np.pi / 2
        grads = np.zeros(len(self.weights))
        base = self._evaluate(data)

        for i, param in enumerate(self.weights):
            # Forward shift
            shift_vector = [0.0] * len(self.weights)
            shift_vector[i] = shift
            fwd = self._evaluate_shift(data, shift_vector)

            # Backward shift
            shift_vector[i] = -shift
            bwd = self._evaluate_shift(data, shift_vector)

            grads[i] = (fwd - bwd).sum() / (2 * shift)
        return grads
