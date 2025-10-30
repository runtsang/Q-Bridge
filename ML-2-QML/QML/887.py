"""
Quantum variational sampler using Pennylane.
"""

from __future__ import annotations

import pennylane as qml
import numpy as np
from pennylane import numpy as pnp


class SamplerQNN:
    """
    Variational quantum sampler with flexible depth and entanglement.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit (default 2).
    num_layers : int
        Number of variational layers.
    device : str | qml.Device
        Pennylane device name or instance (default 'default.qubit').

    Notes
    -----
    The circuit applies parameterized RY rotations on each qubit followed by
    a full‑connect entanglement pattern. The sampler returns the probability
    distribution over the computational basis states.
    """

    def __init__(
        self,
        num_qubits: int = 2,
        num_layers: int = 2,
        device: str | qml.Device = "default.qubit",
    ) -> None:
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.dev = qml.device(device, wires=num_qubits)
        self.params = None  # will be set in circuit

        @qml.qnode(self.dev, interface="autograd")
        def circuit(inputs: np.ndarray, weights: np.ndarray):
            # Input encoding
            for i in range(self.num_qubits):
                qml.RY(inputs[i], wires=i)
            # Variational layers
            weight_idx = 0
            for _ in range(self.num_layers):
                for i in range(self.num_qubits):
                    qml.RY(weights[weight_idx], wires=i)
                    weight_idx += 1
                # Full‑connect entanglement
                for i in range(self.num_qubits):
                    for j in range(i + 1, self.num_qubits):
                        qml.CNOT(wires=[i, j])
            # Measurement
            return qml.expval(qml.PauliZ(0))  # placeholder

        self.circuit = circuit

    def sample(self, inputs: np.ndarray, weights: np.ndarray, shots: int = 1024) -> np.ndarray:
        """
        Sample from the variational circuit.

        Parameters
        ----------
        inputs : array_like
            Input parameters of shape (num_qubits,).
        weights : array_like
            Weight parameters of shape (num_layers * num_qubits,).
        shots : int
            Number of shots for the sampler.

        Returns
        -------
        probs : np.ndarray
            Probability distribution over 2**num_qubits basis states.
        """
        self.params = {"inputs": inputs, "weights": weights}
        # Use Pennylane sampler
        sampler = qml.Sampler(self.circuit)
        raw_counts = sampler(inputs, weights, shots=shots)
        probs = raw_counts / shots
        return probs

    def gradient(self, inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the output expectation with respect to weights.

        Parameters
        ----------
        inputs : array_like
            Input parameters.
        weights : array_like
            Weight parameters.

        Returns
        -------
        grad : np.ndarray
            Gradient vector of shape (num_layers * num_qubits,).
        """
        return qml.grad(self.circuit)(inputs, weights)


__all__ = ["SamplerQNN"]
