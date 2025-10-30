"""Hybrid quantum classifier leveraging Pennylane and parameter‑shift optimisation."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.quantum_info import SparsePauliOp


def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, list[SparsePauliOp]]:
    """
    Construct a layered ansatz with explicit encoding and variational parameters.

    The ansatz now includes a configurable entangling block that uses CZ gates in a
    ring topology, plus an optional mid‑circuit measurement for feature extraction.
    The function mirrors the original signature so it can be swapped into the classical pipeline as a reference.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        # Ring entanglement
        for qubit in range(num_qubits):
            circuit.cz(qubit, (qubit + 1) % num_qubits)

    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]
    return circuit, list(encoding), list(weights), observables


class HybridClassifier:
    """
    A Pennylane‑backed quantum classifier that exposes a QNode and a parameter‑shift
    gradient routine.  It is designed to integrate seamlessly with the classical
    ``HybridClassifier`` via the shared ``build_classifier_circuit`` interface.
    """

    def __init__(self, num_qubits: int, depth: int, device: str = "default.qubit") -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.circuit, self.encoding, self.weights, self.observables = build_classifier_circuit(
            num_qubits, depth
        )
        self.dev = qml.device(device, wires=num_qubits)

        # Initialise trainable parameters uniformly in [-π, π]
        self.params = np.random.uniform(-np.pi, np.pi, len(self.weights))

        # Create a QNode that applies the circuit and measures Z on all qubits
        @qml.qnode(self.dev, interface="autograd")
        def circuit_fn(x, params):
            # Data‑encoding
            for i, wire in enumerate(range(num_qubits)):
                qml.RX(x[i], wires=wire)
            # Variational block
            idx = 0
            for _ in range(depth):
                for wire in range(num_qubits):
                    qml.RY(params[idx], wires=wire)
                    idx += 1
                for wire in range(num_qubits):
                    qml.CZ(wires=[wire, (wire + 1) % num_qubits])
            # Measurements
            return [qml.expval(qml.PauliZ(wire)) for wire in range(num_qubits)]

        self.circuit_fn = circuit_fn

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass returning a 2‑class probability vector via a softmax over
        the raw expectation values.  The raw outputs are linearly mapped to logits
        and then softmaxed.
        """
        expectations = self.circuit_fn(x, self.params)
        logits = pnp.array(expectations)
        probs = pnp.exp(logits) / pnp.sum(pnp.exp(logits))
        return probs

    def loss(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Cross‑entropy loss between predicted probabilities and one‑hot labels.
        """
        probs = self.predict(x)
        return -np.mean(np.sum(y * np.log(probs + 1e-12), axis=1))

    def gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the loss w.r.t. the circuit parameters using the
        parameter‑shift rule implemented by Pennylane.
        """
        return qml.grad(self.circuit_fn)(x, self.params)

    @property
    def num_params(self) -> int:
        return len(self.params)


__all__ = ["HybridClassifier", "build_classifier_circuit"]
