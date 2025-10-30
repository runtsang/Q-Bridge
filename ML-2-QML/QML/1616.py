import pennylane as qml
import numpy as np

class QuantumConvolutionFilter:
    """
    Quantum version of a convolutional filter.
    Maps a kernel of pixel values to the average probability of measuring |1>
    across all qubits. Suitable for hybrid classical‑quantum pipelines.
    """

    def __init__(self,
                 kernel_size: int = 2,
                 threshold: float = 0.0,
                 shots: int = 1024,
                 backend: str = "default.qubit"):
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.shots = shots
        self.n_qubits = kernel_size ** 2

        dev = qml.device(backend, wires=self.n_qubits, shots=self.shots)

        @qml.qnode(dev)
        def circuit(pixel_values):
            for i in range(self.n_qubits):
                theta = np.pi if pixel_values[i] > self.threshold else 0.0
                qml.RX(theta, wires=i)
            # simple entangling layer
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            # return expectation values of PauliZ for each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self._circuit = circuit

    def run(self, data: np.ndarray) -> float:
        """
        Run the quantum circuit on a 2‑D array of pixel values.

        Parameters
        ----------
        data : np.ndarray
            2‑D array of shape (kernel_size, kernel_size).

        Returns
        -------
        float
            Average probability of measuring |1> across all qubits.
        """
        flat = data.reshape(self.n_qubits)
        expz = self._circuit(flat)  # list of expectation values of PauliZ
        # Convert expectation values to probability of |1>
        p1 = [(1 - e) / 2.0 for e in expz]
        return np.mean(p1)
