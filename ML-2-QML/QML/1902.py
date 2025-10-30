"""
Quantum sampler network using PennyLane with a full variational circuit.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


def SamplerQNN() -> object:
    """
    Returns an instance of the updated SamplerQNN__gen088 quantum module.
    The circuit encodes two input angles, applies entangling CZ gates, and
    uses four trainable rotation angles. It outputs a 4‑dimensional probability
    vector over the two‑qubit computational basis, which is then mapped to
    a 2‑class distribution by summing the first two basis states.
    The class exposes:
    - ``forward``: compute the softmax‑like output probabilities.
    - ``sample``: draw samples from the output distribution.
    - ``set_params`` / ``get_params``: parameter management.
    """
    class SamplerQNN__gen088:
        def __init__(self, device_name: str = "default.qubit", shots: int = 1000):
            self.dev = qml.device(device_name, wires=2, shots=shots)
            # Initialize input and weight parameters
            self.params = np.random.uniform(0, 2 * np.pi, (2, 4))
            self._build_circuit()

        def _build_circuit(self) -> None:
            @qml.qnode(self.dev, interface="autograd")
            def circuit(inputs, weights):
                # Input encoding
                qml.RY(inputs[0], wires=0)
                qml.RY(inputs[1], wires=1)
                # Entanglement
                qml.CZ(wires=[0, 1])
                # Parameterized rotations
                qml.RY(weights[0], wires=0)
                qml.RY(weights[1], wires=1)
                qml.CZ(wires=[0, 1])
                qml.RY(weights[2], wires=0)
                qml.RY(weights[3], wires=1)
                # Return full probability vector over 2 qubits
                return qml.probs(wires=[0, 1])
            self.circuit = circuit

        def forward(self, inputs: np.ndarray) -> np.ndarray:
            """
            Compute the 2‑class probability distribution from the quantum circuit.
            Parameters
            ----------
            inputs : np.ndarray
                Input array of shape (2,) or (batch, 2).
            Returns
            -------
            np.ndarray
                Probability vector of shape (2,) or (batch, 2).
            """
            probs = self.circuit(inputs, self.params)
            # Map the 4‑dimensional probability vector to 2 classes
            # For simplicity, use the first two basis states as the outputs
            return probs[:, :2] if probs.ndim > 1 else probs[:2]

        def sample(self, inputs: np.ndarray, num_samples: int = 100) -> np.ndarray:
            """
            Draw discrete samples from the output distribution.
            Parameters
            ----------
            inputs : np.ndarray
                Input array of shape (2,) or (batch, 2).
            num_samples : int
                Number of samples to draw per input.
            Returns
            -------
            np.ndarray
                Sampled class indices of shape (num_samples,) or (batch, num_samples).
            """
            probs = self.forward(inputs)
            if probs.ndim == 1:
                return np.random.choice(2, size=num_samples, p=probs)
            return np.array([np.random.choice(2, size=num_samples, p=p) for p in probs])

        def set_params(self, new_params: np.ndarray) -> None:
            """
            Replace the current trainable parameters.
            """
            self.params = new_params

        def get_params(self) -> np.ndarray:
            """
            Retrieve the current trainable parameters.
            """
            return self.params

    return SamplerQNN__gen088()
