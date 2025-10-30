"""
Quantum sampler network using PennyLane.

This module defines `SamplerQNN`, a variational quantum circuit
with two qubits that accepts classical inputs and trainable
weights.  The circuit includes a full‑entanglement layer
and uses expectation values of Pauli‑Z to produce a two‑dimensional
probability distribution via a classical soft‑max layer.
The class exposes a `sample` method that returns sampled indices
and a `probabilities` method that returns the full distribution.
"""

from __future__ import annotations

import pennylane as qml
import numpy as np
from typing import Tuple

class SamplerQNN:
    """
    Variational quantum sampler.

    Parameters
    ----------
    dev : qml.Device
        PennyLane quantum device (default: 'default.qubit' with 2 wires).
    num_qubits : int
        Number of qubits in the circuit (default: 2).
    entanglement : str
        Entanglement scheme for the rotation layers ('full', 'pairwise', etc.).
    """

    def __init__(
        self,
        dev: qml.Device | None = None,
        num_qubits: int = 2,
        entanglement: str = "full",
    ) -> None:
        self.num_qubits = num_qubits
        self.entanglement = entanglement
        self.dev = dev or qml.device("default.qubit", wires=num_qubits)

        # Parameter vector: first 2 are classical inputs, remaining are trainable weights
        self.param_len = 2 + 4  # 2 inputs + 4 weights for a 2‑qubit circuit

        @qml.qnode(self.dev, interface="torch")
        def circuit(params: qml.numpy.ndarray) -> Tuple[float, float]:
            """Parameterized quantum circuit."""
            # Classical inputs as rotation angles
            qml.RY(params[0], wires=0)
            qml.RY(params[1], wires=1)

            # Entanglement layer
            for i in range(num_qubits):
                for j in range(i + 1, num_qubits):
                    qml.CNOT(wires=[i, j])

            # Trainable rotation layer
            for w in range(2, self.param_len):
                qml.RY(params[w], wires=(w - 2) % num_qubits)

            # Measure expectation of Z on each qubit
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

        self._circuit = circuit

    def probabilities(self, inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Compute the probability distribution over the two outputs.

        Parameters
        ----------
        inputs : array-like, shape (2,)
            Classical input angles.
        weights : array-like, shape (4,)
            Trainable rotation angles.

        Returns
        -------
        probs : ndarray, shape (2,)
            Soft‑max probabilities.
        """
        params = np.concatenate([inputs, weights])
        exp_vals = self._circuit(params)
        # Convert expectation values to probabilities via a sigmoid mapping
        probs_raw = 0.5 * (np.array(exp_vals) + 1.0)
        # Normalize to ensure sum == 1
        probs = probs_raw / probs_raw.sum()
        return probs

    def sample(self, inputs: np.ndarray, weights: np.ndarray, num_samples: int = 1) -> np.ndarray:
        """
        Draw samples from the quantum sampler.

        Parameters
        ----------
        inputs : array-like, shape (2,)
            Classical input angles.
        weights : array-like, shape (4,)
            Trainable rotation angles.
        num_samples : int
            Number of samples to draw.

        Returns
        -------
        samples : ndarray, shape (num_samples,)
            Sample indices (0 or 1) drawn according to the probability distribution.
        """
        probs = self.probabilities(inputs, weights)
        return np.random.choice(len(probs), size=num_samples, p=probs)

__all__ = ["SamplerQNN"]
