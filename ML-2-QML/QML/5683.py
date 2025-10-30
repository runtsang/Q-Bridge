"""Quantum convolution filter using Pennylane.

The :class:`QuanConv` class implements a parameterised variational circuit that
acts on a ``kernel_size × kernel_size`` patch.  It supports batch evaluation,
dynamic thresholding, and can be combined with the classical :class:`ConvFilter`
for hybrid inference.  The circuit is compiled for the Qiskit Aer simulator
by default but can be swapped for any Pennylane device.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

__all__ = ["QuanConv"]


class QuanConv:
    """
    Variational quantum filter for a convolutional kernel.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the kernel (square).  The number of qubits equals ``kernel_size**2``.
    shots : int, default 1024
        Number of shots for each evaluation.
    device : str or pennylane.Device, default "default.qubit"
        Pennylane device name or instance.
    threshold : float, default 0.0
        Classical threshold used to decide whether a qubit is in |1> state.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        shots: int = 1024,
        device: str | qml.Device = "default.qubit",
        threshold: float = 0.0,
    ) -> None:
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.shots = shots
        self.threshold = threshold

        if isinstance(device, str):
            self.dev = qml.device(device, wires=self.n_qubits, shots=self.shots)
        else:
            self.dev = device

        # Parameterised angles for each qubit (one per qubit)
        self.params = pnp.random.uniform(0, 2 * np.pi, self.n_qubits)

        # Build the device‑agnostic circuit
        @qml.qnode(self.dev, interface="autograd")
        def circuit(x, params):
            # Encode classical data into rotation angles
            for i, val in enumerate(x):
                theta = np.pi if val > self.threshold else 0.0
                qml.RX(theta, wires=i)
            # Variational layer
            for i in range(self.n_qubits):
                qml.RY(params[i], wires=i)
            # Measure
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.circuit = circuit

    def run(self, data: np.ndarray) -> float:
        """
        Run the quantum filter on a single kernel patch.

        Parameters
        ----------
        data : np.ndarray
            Array of shape ``(kernel_size, kernel_size)`` with pixel values.

        Returns
        -------
        float
            Average probability of measuring |1> across all qubits.
        """
        flat = data.reshape(self.n_qubits)
        meas = self.circuit(flat, self.params)
        probs = (np.array(meas) + 1) / 2  # convert expectation value to probability
        return probs.mean().item()

    def batch_run(self, batch: np.ndarray) -> np.ndarray:
        """
        Evaluate a batch of kernel patches.

        Parameters
        ----------
        batch : np.ndarray
            Shape ``(batch_size, kernel_size, kernel_size)``.

        Returns
        -------
        np.ndarray
            Shape ``(batch_size,)`` with average |1> probabilities.
        """
        outputs = []
        for patch in batch:
            outputs.append(self.run(patch))
        return np.array(outputs)

    def hybrid_forward(self, data: np.ndarray, classical_filter: ConvFilter) -> np.ndarray:
        """
        Hybrid inference that blends classical sigmoid activations with
        quantum measurement statistics.

        Parameters
        ----------
        data : np.ndarray
            Shape ``(batch, 1, H, W)`` of grayscale images.
        classical_filter : ConvFilter
            Instance of the classical filter to use for the first stage.

        Returns
        -------
        np.ndarray
            Shape ``(batch,)`` with blended probabilities.
        """
        # Classical part
        class_probs = classical_filter(
            torch.as_tensor(data, dtype=torch.float32)
        ).detach().numpy()

        # Quantum part
        patches = self._extract_patches(data, self.kernel_size)
        quantum_probs = self.batch_run(patches)

        # Simple linear blending
        return 0.5 * class_probs + 0.5 * quantum_probs

    @staticmethod
    def _extract_patches(data: np.ndarray, k: int) -> np.ndarray:
        """
        Utility to extract sliding patches from an image batch.
        """
        batch, _, H, W = data.shape
        patches = []
        for i in range(H - k + 1):
            for j in range(W - k + 1):
                patch = data[:, :, i : i + k, j : j + k]
                patches.append(patch.reshape(batch, -1))
        return np.stack(patches, axis=1)
