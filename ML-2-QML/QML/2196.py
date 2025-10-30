"""Quantum convolution module using Pennylane.

This implementation extends the original Qiskit‑based quanvolution by
- replacing the low‑level circuit construction with a variational
  feature‑map that encodes image patches as qubit rotations.
- providing a concise qnode that can be executed on either a simulator
  or a real device via Pennylane.
- exposing a ``run`` method that returns the probability of measuring
  |1> on a randomly chosen qubit, averaged over the whole filter.
"""

from __future__ import annotations

import pennylane as qml
import numpy as np
from typing import Optional


class ConvQuantum:
    """
    Quantum convolutional filter implemented with Pennylane.

    Parameters
    ----------
    kernel_size : int, optional
        Size of the square filter (default 2).
    threshold : float, optional
        Threshold used to binarize the image patch before encoding
        (default 127).
    shots : int, optional
        Number of shots for the simulator (default 100).
    device : str | qml.Device, optional
        Pennylane device name or instance (default "default.qubit").
    """

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 127,
        shots: int = 100,
        device: str | qml.Device = "default.qubit",
    ) -> None:
        self.n_qubits = kernel_size ** 2
        self.threshold = threshold
        self.shots = shots
        if isinstance(device, str):
            self.dev = qml.device(device, wires=self.n_qubits, shots=self.shots)
        else:
            self.dev = device
        self._circuit = self._build_circuit()

    def _build_circuit(self):
        @qml.qnode(self.dev, interface="autograd")
        def circuit(flat_data):
            # Encode each pixel as a Ry rotation
            for i, val in enumerate(flat_data):
                qml.RY(np.pi * (val > self.threshold), wires=i)

            # Entangling feature map
            for i in range(self.n_qubits):
                qml.CNOT(wires=[i, (i + 1) % self.n_qubits])

            # Expectation value of Z on a randomly chosen qubit
            return qml.expval(qml.PauliZ(wires=np.random.randint(self.n_qubits)))

        return circuit

    def run(self, data: np.ndarray) -> float:
        """
        Execute the quantum filter on a single image patch.

        Parameters
        ----------
        data : np.ndarray
            2‑D array of shape ``(kernel_size, kernel_size)``.

        Returns
        -------
        float
            Estimated probability of measuring |1> on the chosen qubit.
        """
        flat = data.reshape(-1)
        exp_val = self._circuit(flat)
        # Convert expectation <Z> to probability of |1>
        prob_1 = 0.5 * (1 - exp_val)
        return float(prob_1)

    def run_batch(self, batch: np.ndarray) -> list[float]:
        """
        Execute the filter on a batch of patches.

        Parameters
        ----------
        batch : np.ndarray
            Array of shape ``(B, kernel_size, kernel_size)``.

        Returns
        -------
        list[float]
            Probabilities for each patch in the batch.
        """
        return [self.run(patch) for patch in batch]
