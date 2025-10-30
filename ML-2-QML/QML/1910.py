"""Quantum convolutional filter using a variational circuit.

The circuit encodes a 2D kernel as a binary pattern via RX rotations (π if the pixel value
exceeds a learnable threshold).  A single variational layer of RY gates followed by a
chain of CNOTs entangles the qubits.  The circuit returns the mean probability of
measuring |1> across all qubits, matching the interface of the classical implementation.

Example usage:

>>> from Conv__gen097 import Conv as QConv
>>> q_conv = QConv()
>>> q_conv.run(np.random.rand(2,2))
0.48
"""

import numpy as np
import pennylane as qml

class QuanvCircuit:
    """
    Variational quanvolution circuit with a binary data encoding and a trainable
    RY layer.  The circuit is fully differentiable with Pennylane, allowing
    gradient‑based training of the parameters via a PyTorch or JAX backend.

    Parameters
    ----------
    kernel_size : int, optional
        Size of the square kernel (default 2).
    device : pennylane.Device, optional
        Quantum device.  If None, a default.qubit simulator is used.
    threshold : float, optional
        Threshold used for binary data encoding.  The value of each pixel is
        compared against this threshold to decide whether an RX(π) gate is
        applied.  The threshold can also be treated as a trainable
        parameter if desired.
    shots : int, optional
        Number of shots for the simulator.  Default is 1024.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        device: qml.Device | None = None,
        threshold: float = 0.5,
        shots: int = 1024,
    ) -> None:
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.threshold = threshold
        self.shots = shots
        self.device = device or qml.device("default.qubit", wires=self.n_qubits, shots=shots)
        # Trainable parameters for the variational RY layer
        self.params = np.random.randn(self.n_qubits)

        @qml.qnode(self.device, interface="autograd")
        def circuit(data, params):
            # Binary encoding of the input data
            for i in range(self.n_qubits):
                angle = np.pi if data[i] > self.threshold else 0.0
                qml.RX(angle, wires=i)
            # Variational RY layer
            for i in range(self.n_qubits):
                qml.RY(params[i], wires=i)
            # Simple chain entanglement
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            # Return probabilities for each qubit
            return qml.probs(wires=range(self.n_qubits))

        self.circuit = circuit

    def run(self, data: np.ndarray) -> float:
        """
        Execute the circuit on a 2‑D kernel and return the average probability
        of measuring |1> across all qubits.

        Parameters
        ----------
        data : array‑like
            Kernel with shape (kernel_size, kernel_size).

        Returns
        -------
        float
            Average |1> probability.
        """
        data = np.reshape(data, (self.n_qubits,))
        probs = self.circuit(data, self.params)
        # probs shape: (n_qubits, 2)
        ones_prob = probs[:, 1].mean().item()
        return ones_prob

def Conv() -> QuanvCircuit:
    """Factory that returns a default 2×2 quanvolution circuit."""
    return QuanvCircuit(kernel_size=2, threshold=0.5, shots=1024)

__all__ = ["Conv"]
