"""
HybridSamplerQNN – Quantum implementation
========================================

This module implements a variational quantum sampler using Pennylane.
The circuit consists of multiple parameterized rotation layers and
entangling CNOTs, and it outputs a probability distribution over
the computational basis.  The implementation is fully compatible with
Pennylane's `qml.QNode` and supports state‑vector sampling for
exact probabilities or shot‑based sampling for approximate results.
"""

from __future__ import annotations

import pennylane as qml
import numpy as np
from pennylane import numpy as pnp
from typing import Sequence


class HybridSamplerQNN:
    """
    A variational quantum sampler.

    Parameters
    ----------
    num_wires : int, default 2
        Number of qubits / input dimensions.
    num_layers : int, default 2
        Number of parameterized rotation layers.
    device_name : str, default 'default.qubit'
        Pennylane device to use.
    """

    def __init__(
        self,
        num_wires: int = 2,
        num_layers: int = 2,
        device_name: str = "default.qubit",
    ) -> None:
        self.num_wires = num_wires
        self.num_layers = num_layers
        self.dev = qml.device(device_name, wires=num_wires, shots=None)
        self.params = np.zeros((num_layers, num_wires, 3), dtype=np.float64)

        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs: np.ndarray, params: np.ndarray) -> np.ndarray:
            # Encode the inputs via Ry rotations
            for i in range(num_wires):
                qml.RY(inputs[i], wires=i)
            # Variational layers
            for layer in range(num_layers):
                for i in range(num_wires):
                    qml.RY(params[layer, i, 0], wires=i)
                    qml.RZ(params[layer, i, 1], wires=i)
                # Entanglement
                for i in range(num_wires - 1):
                    qml.CNOT(wires=[i, i + 1])
            # Measurement: return probabilities of computational basis
            return qml.probs(wires=range(num_wires))

        self.circuit = circuit

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Evaluate the circuit and return a probability distribution.

        Parameters
        ----------
        inputs : np.ndarray
            Input vector of shape (num_wires,).

        Returns
        -------
        np.ndarray
            Probabilities over 2^num_wires basis states.
        """
        return self.circuit(inputs, self.params)

    def sample(self, inputs: np.ndarray, shots: int = 1024) -> np.ndarray:
        """
        Sample from the circuit using a finite number of shots.

        Parameters
        ----------
        inputs : np.ndarray
            Input vector of shape (num_wires,).
        shots : int, default 1024
            Number of measurement shots.

        Returns
        -------
        np.ndarray
            Sampled counts for each basis state.
        """
        # Re‑create device with shots for sampling
        self.dev = qml.device("default.qubit", wires=self.num_wires, shots=shots)

        @qml.qnode(self.dev, interface="torch")
        def sampling_circuit(inputs: np.ndarray, params: np.ndarray) -> np.ndarray:
            for i in range(self.num_wires):
                qml.RY(inputs[i], wires=i)
            for layer in range(self.num_layers):
                for i in range(self.num_wires):
                    qml.RY(params[layer, i, 0], wires=i)
                    qml.RZ(params[layer, i, 1], wires=i)
                for i in range(self.num_wires - 1):
                    qml.CNOT(wires=[i, i + 1])
            return qml.sample(wires=range(self.num_wires))

        return sampling_circuit(inputs, self.params)

    def parameters(self) -> np.ndarray:
        """Return the trainable parameters."""
        return self.params

    def set_parameters(self, new_params: np.ndarray) -> None:
        """Set the trainable parameters."""
        assert new_params.shape == self.params.shape
        self.params = new_params


__all__ = ["HybridSamplerQNN"]
