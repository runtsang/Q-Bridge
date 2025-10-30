"""ConvEnhanced: a quantum variational filter that emulates a convolutional kernel."""

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


class ConvEnhanced:
    """
    Quantum variational filter that emulates a convolutional kernel.
    The circuit applies a data‑dependent rotation followed by a parameterised
    entangling layer and a point‑wise rotation on each qubit.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        shots: int = 100,
        threshold: float = 0.5,
        device: qml.Device | None = None,
    ) -> None:
        """
        Parameters
        ----------
        kernel_size : int
            Size of the square kernel.
        shots : int
            Number of shots for the simulator.
        threshold : float
            Data threshold used to decide the RX rotation.
        device : pennylane.Device, optional
            Quantum device; defaults to `default.qubit`.
        """
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size**2
        self.threshold = threshold
        self.shots = shots
        self.dev = device or qml.device("default.qubit", wires=self.n_qubits, shots=self.shots)
        self.qnode = qml.QNode(self._circuit, self.dev)

    def _circuit(self, params, data):
        # data is a flattened array of shape (n_qubits,)
        for i in range(self.n_qubits):
            if data[i] > self.threshold:
                qml.RX(np.pi, wires=i)
            else:
                qml.RX(0.0, wires=i)

        # entanglement layer
        for i in range(self.n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])

        # parameterised rotations
        for i in range(self.n_qubits):
            qml.Rot(params[i, 0], params[i, 1], params[i, 2], wires=i)

        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def run(self, data, params=None) -> float:
        """
        Execute the circuit and return a scalar activation.

        Parameters
        ----------
        data : array‑like
            2‑D array with shape (kernel_size, kernel_size).
        params : array‑like, optional
            Parameter matrix of shape (n_qubits, 3).  If None, zeros are used.
            The returned value can be used as part of a differentiable loss.

        Returns
        -------
        float
            Mean probability of measuring |1> across all qubits.
        """
        data_flat = np.reshape(data, (self.n_qubits,))
        if params is None:
            params = np.zeros((self.n_qubits, 3), requires_grad=True)

        expvals = self.qnode(params, data_flat)
        probs = (1 - np.array(expvals)) / 2
        return probs.mean()
