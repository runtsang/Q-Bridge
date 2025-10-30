import pennylane as qml
import numpy as np

class Conv:
    """
    Quantum convolution filter using a Pennylane variational circuit.
    The filter acts on a kernel‑sized patch of an image. The circuit
    encodes the patch into rotation angles and applies a depth‑wise
    entangling layer. The output is the average probability of measuring
    the |1> state across all qubits. This can be used as a drop‑in
    replacement for the classical Conv class.
    """

    def __init__(self,
                 kernel_size: int = 2,
                 depth: int = 1,
                 shots: int = 1000,
                 threshold: float = 0.5,
                 device: str = 'default.qubit'):
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.depth = depth
        self.shots = shots
        self.threshold = threshold
        self.dev = qml.device(device, wires=self.n_qubits, shots=shots)

        # Parameter vector for the variational layer
        self.theta = np.random.uniform(0, 2*np.pi, size=(depth, self.n_qubits))

        @qml.qnode(self.dev)
        def circuit(data, theta):
            # Encode data as Ry rotations
            for i in range(self.n_qubits):
                qml.RY(data[i] * np.pi, wires=i)
            # Variational layers
            for d in range(depth):
                for i in range(self.n_qubits):
                    qml.RY(theta[d, i], wires=i)
                # Entangling layer: nearest‑neighbor CNOT chain
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            # Expectation of PauliZ on each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.circuit = circuit

    def run(self, data: np.ndarray) -> float:
        """
        Run the quantum filter on a single kernel patch.

        Parameters
        ----------
        data : np.ndarray
            2D array of shape (kernel_size, kernel_size) with pixel values in
            [0, 1].

        Returns
        -------
        float
            Average probability of measuring |1> across all qubits.
        """
        # Binarize data according to threshold
        binary = (data > self.threshold).astype(np.float32)
        flat = binary.flatten()
        z_expvals = self.circuit(flat, self.theta)
        # Convert PauliZ expectation to probability of |1>
        probs = (1 - np.array(z_expvals)) / 2
        return probs.mean()

    def set_threshold(self, value: float) -> None:
        """Update the threshold used for data preprocessing."""
        self.threshold = value

    def set_theta(self, new_theta: np.ndarray) -> None:
        """Replace the variational parameters."""
        self.theta = new_theta
