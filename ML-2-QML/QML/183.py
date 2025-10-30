"""
Quantum sampler network using Pennylane.

Features:
* Three‑qubit variational circuit with rotation and entangling layers.
* Parameter vector split into input and weight parameters.
* Built‑in sampling via Pennylane's QuantumDevice.
* Exposes a `sample` method returning measurement outcomes.
"""

from __future__ import annotations

import pennylane as qml
import numpy as np
from pennylane import numpy as pnp
from typing import Tuple


class SamplerQNN:
    """
    Variational quantum sampler.

    Parameters
    ----------
    n_qubits : int, default 3
        Number of qubits in the circuit.
    entanglement : str, default "full"
        Entanglement pattern for the entangling layers.
    device_name : str, default "default.qubit"
        Pennylane device to use for simulation or execution.
    """

    def __init__(
        self,
        n_qubits: int = 3,
        entanglement: str = "full",
        device_name: str = "default.qubit",
    ) -> None:
        self.n_qubits = n_qubits
        self.device = qml.device(device_name, wires=n_qubits)
        self.entanglement = entanglement
        # Parameter register: first 2 are inputs, remaining 2 * n_qubits are weights
        self.input_params = pnp.arange(2)
        self.weight_params = pnp.arange(2 * n_qubits) + 2

        @qml.qnode(self.device, interface="autograd")
        def circuit(inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
            # Input rotations
            for i in range(n_qubits):
                qml.RY(inputs[i], wires=i)
            # Entangling layer
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            # Parameterised rotations
            for i in range(n_qubits):
                qml.RY(weights[i], wires=i)
            # Second entangling layer
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            # Final rotations
            for i in range(n_qubits):
                qml.RY(weights[n_qubits + i], wires=i)
            return qml.expval(qml.PauliZ(0))

        self._circuit = circuit

    def forward(self, inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Evaluate the circuit and return the expectation value.

        Parameters
        ----------
        inputs : np.ndarray
            Input parameters of shape ``(n_qubits,)``.
        weights : np.ndarray
            Weight parameters of shape ``(2 * n_qubits,)``.

        Returns
        -------
        np.ndarray
            Expectation value of Pauli‑Z on qubit 0.
        """
        return self._circuit(inputs, weights)

    def sample(
        self,
        inputs: np.ndarray,
        weights: np.ndarray,
        n_shots: int = 1024,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Draw measurement samples from the variational circuit.

        Parameters
        ----------
        inputs : np.ndarray
            Input parameters.
        weights : np.ndarray
            Weight parameters.
        n_shots : int, default 1024
            Number of measurement shots.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (counts, probabilities) of measuring |0⟩ or |1⟩ on qubit 0.
        """
        self.device.reset()
        # Perform shots measurement
        probs = self.device.execute(self._circuit, shots=n_shots, return_counts=True)
        # Simplify counts to two outcomes
        counts = {k: v for k, v in probs.items() if k in ["0", "1"]}
        probs_arr = np.array([counts.get("0", 0), counts.get("1", 0)]) / n_shots
        return np.array(list(counts.values())), probs_arr

__all__ = ["SamplerQNN"]
