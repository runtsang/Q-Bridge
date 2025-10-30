"""
Quantum sampler network with a variational circuit.
"""

from __future__ import annotations

import pennylane as qml
import pennylane.numpy as np
from pennylane import numpy as penn_np
from pennylane import QNode
from pennylane import Device
from pennylane.qnodes import QuantumNode
from pennylane.optimize import AdamOptimizer


class SamplerQNN:
    """
    Variational quantum sampler that outputs a probability distribution over 2 qubits.
    The circuit uses a two‑layer entangling ansatz and can be sampled or optimized.
    """

    def __init__(self, num_qubits: int = 2, layers: int = 2, dev: Device | None = None) -> None:
        self.num_qubits = num_qubits
        self.layers = layers
        self.dev = dev or qml.device("default.qubit", wires=num_qubits)
        self.weights = np.random.randn(layers, num_qubits, 3)  # RX,RZ,RY per qubit per layer
        self.input_params = np.random.randn(num_qubits)  # placeholder for classical inputs

        @qml.qnode(self.dev, interface="autograd")
        def circuit(inputs, weights):
            # Encode classical inputs as rotations
            for i in range(num_qubits):
                qml.RY(inputs[i], wires=i)
            # Variational layers
            for layer in range(layers):
                for qubit in range(num_qubits):
                    qml.RX(weights[layer, qubit, 0], wires=qubit)
                    qml.RZ(weights[layer, qubit, 1], wires=qubit)
                    qml.RY(weights[layer, qubit, 2], wires=qubit)
                # Entangling via CX
                for qubit in range(num_qubits - 1):
                    qml.CNOT(wires=[qubit, qubit + 1])
            return qml.probs(wires=range(num_qubits))

        self._circuit = circuit

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Evaluate the circuit and return the probability distribution.
        """
        return self._circuit(inputs, self.weights)

    def sample(self, inputs: np.ndarray, shots: int = 1000) -> np.ndarray:
        """
        Sample measurement outcomes from the circuit.

        Parameters
        ----------
        inputs : np.ndarray
            Classical input vector of shape (num_qubits,).
        shots : int
            Number of measurement shots.

        Returns
        -------
        samples : np.ndarray
            Array of shape (shots, num_qubits) with sampled bitstrings.
        """
        probs = self.forward(inputs)
        return qml.measurement.sample(probs, shots=shots)

    def train(self, data: np.ndarray, epochs: int = 200, lr: float = 0.01) -> None:
        """
        Train the variational parameters to match target distribution using Adam.

        Parameters
        ----------
        data : np.ndarray
            Target samples of shape (N, num_qubits).
        epochs : int
            Training steps.
        lr : float
            Learning rate.
        """
        optimizer = AdamOptimizer(lr)
        for _ in range(epochs):
            def loss_fn(weights):
                probs = self._circuit(self.input_params, weights)
                target = np.zeros_like(probs)
                for sample in data:
                    idx = int("".join(str(bit) for bit in sample), 2)
                    target[idx] += 1
                target /= len(data)
                return -np.sum(target * np.log(probs + 1e-10))
            self.weights = optimizer.step(loss_fn, self.weights)

__all__ = ["SamplerQNN"]
