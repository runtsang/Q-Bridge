"""Variational quantum filter using Pennylane."""
import pennylane as qml
import numpy as np


class ConvFusion:
    """
    Quantum‑enhanced filter that encodes a 2‑D patch into a variational
    circuit of ``kernel_size**2`` qubits.  The circuit consists of
    data‑encoding rotations followed by a user‑defined number of
    parameter‑tunable layers.  The output is the average expectation
    value of Pauli‑Z on all qubits.

    The implementation is fully differentiable via Pennylane’s
    automatic differentiation, making it suitable for hybrid
    training pipelines.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        num_layers: int = 2,
        dev: qml.Device | None = None,
    ) -> None:
        """
        Parameters
        ----------
        kernel_size
            Size of the square input patch.
        num_layers
            Number of variational layers after data encoding.
        dev
            Pennylane device; defaults to the in‑memory simulator.
        """
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.num_layers = num_layers
        self.dev = dev or qml.device("default.qubit", wires=self.n_qubits)

        # Initialize trainable parameters
        self.params = np.random.randn(self.num_layers, self.n_qubits)

        @qml.qnode(self.dev, interface="numpy")
        def circuit(x: np.ndarray, params: np.ndarray):
            # Data encoding: rotation about Y proportional to pixel intensity
            for i in range(self.n_qubits):
                qml.RY(np.pi * x[i], wires=i)

            # Variational layers
            for l in range(self.num_layers):
                for i in range(self.n_qubits):
                    qml.RY(params[l, i], wires=i)
                # Entanglement (ring topology)
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[self.n_qubits - 1, 0])

            # Return expectation values of Pauli‑Z on all qubits
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.circuit = circuit

    def run(self, data) -> float:
        """
        Execute the quantum filter on a 2‑D patch.

        Parameters
        ----------
        data
            2‑D array with shape (kernel_size, kernel_size) and
            values in the range [0, 255].

        Returns
        -------
        float
            Average expectation value across all qubits (range [−1, 1]).
        """
        # Flatten and normalize pixel intensities to [0, 1]
        x = np.reshape(data, -1) / 255.0
        expvals = self.circuit(x, self.params)
        return float(np.mean(expvals))


__all__ = ["ConvFusion"]
