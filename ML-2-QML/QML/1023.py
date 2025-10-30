"""ConvQuantum: a lightweight variational filter for hybrid convolution."""

import numpy as np
import pennylane as qml
import pennylane.numpy as pnp
from typing import Tuple


class ConvQuantum:
    """
    Variational circuit that maps a 2‑D patch to a scalar activation.
    The circuit is parameterized by a rotation angle per qubit, which is
    set based on the input pixel intensity relative to a threshold.
    """

    def __init__(self, kernel_size: int = 2, shots: int = 1000, backend: str = "default.qubit"):
        self.n_qubits = kernel_size ** 2
        self.shots = shots
        self.dev = qml.device(backend, wires=self.n_qubits)

        @qml.qnode(self.dev, interface="autograd")
        def circuit(params: pnp.ndarray) -> pnp.ndarray:
            for i in range(self.n_qubits):
                qml.RX(params[i], wires=i)
            # Entangle neighboring qubits in a ring
            for i in range(self.n_qubits):
                qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.circuit = circuit

    @staticmethod
    def _threshold_to_angle(value: float, threshold: float) -> float:
        """
        Map a pixel value to a rotation angle.
        Values above the threshold are rotated by π, otherwise 0.
        """
        return np.pi if value > threshold else 0.0

    def run(self, data: np.ndarray, threshold: float) -> float:
        """
        Run the variational circuit on a 2‑D patch.

        Args:
            data: 2‑D array with shape (kernel_size, kernel_size).
            threshold: float threshold for setting rotation angles.

        Returns:
            float: average probability of measuring |1> across qubits.
        """
        flat = data.flatten()
        params = np.array([self._threshold_to_angle(v, threshold) for v in flat])
        # Execute the circuit
        expvals = self.circuit(params)
        # Convert expectation values to probabilities of |1>
        probs = (1 - np.array(expvals)) / 2
        return probs.mean()

    @classmethod
    def apply(cls, x: torch.Tensor, threshold: float) -> torch.Tensor:
        """
        Class method to be called from the classical ConvEnhanced module.
        Converts a torch tensor to numpy, runs the quantum filter, and returns
        a torch scalar tensor.
        """
        # x is expected to have shape (batch, 1, H, W)
        batch = x.shape[0]
        out = []
        for i in range(batch):
            patch = x[i, 0].detach().cpu().numpy()
            # Assume patch is square and its size matches the circuit's n_qubits
            conv = cls(kernel_size=int(np.sqrt(patch.size)), shots=100)
            out.append(conv.run(patch, threshold))
        return torch.tensor(out, device=x.device)


__all__ = ["ConvQuantum"]
