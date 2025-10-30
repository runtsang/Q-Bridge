import numpy as np
import pennylane as qml
import pennylane.numpy as pnp

class ConvFilterHybrid:
    """Quantum-only convolutional filter inspired by quanvolution.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the filter kernel.
    shots : int, default 100
        Number of shots for the quantum backend.
    threshold : float, default 127
        Threshold for binarizing pixel values before encoding.
    """
    def __init__(self, kernel_size: int = 2, shots: int = 100, threshold: float = 127) -> None:
        self.kernel_size = kernel_size
        self.shots = shots
        self.threshold = threshold
        self.n_qubits = kernel_size ** 2
        self.dev = qml.device("default.qubit", wires=self.n_qubits)

        # Build a simple random circuit as in the seed
        self.circuit = self._build_circuit()

    def _build_circuit(self):
        @qml.qnode(self.dev, interface="numpy")
        def circuit(x):
            # Encode data into RX rotations
            for i, val in enumerate(x):
                angle = np.pi if val > self.threshold else 0.0
                qml.RX(angle, wires=i)
            # Add a small random ansatz
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.RZ(np.pi / 4, wires=0)
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        return circuit

    def run(self, data: np.ndarray) -> float:
        """Execute the quantum filter on a 2D kernel array.

        Returns
        -------
        float
            Average probability of measuring |1> across qubits.
        """
        x_flat = data.reshape(-1)
        outputs = self.circuit(x_flat)
        # Convert expectation values to probabilities
        probs = (1 - np.array(outputs)) / 2
        return probs.mean()
