import torch
import pennylane as qml
import pennylane.numpy as np
from pennylane import qnode

class QuantumKernelMethod:
    """Quantum kernel using PennyLane with a simple RY‑CNOT ansatz."""
    def __init__(self, num_qubits: int = 4, dev: qml.Device | None = None):
        self.num_qubits = num_qubits
        self.dev = dev or qml.device("default.qubit", wires=num_qubits)
        self._build_ansatz()

    def _build_ansatz(self):
        @qml.qnode(self.dev, interface="torch")
        def circuit(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            # Encode the first vector
            for i in range(self.num_qubits):
                qml.RY(x[i], wires=i)
            # Entangle
            for i in range(self.num_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            # Encode the second vector with a sign flip
            for i in range(self.num_qubits):
                qml.RY(-y[i], wires=i)
            return qml.expval(qml.PauliZ(0))
        self.circuit = circuit

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the absolute value of the kernel evaluation."""
        return torch.abs(self.circuit(x, y))

    def kernel_matrix(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Compute the Gram matrix between two batches of samples."""
        n, m = a.shape[0], b.shape[0]
        K = torch.empty((n, m), device=a.device)
        for i in range(n):
            for j in range(m):
                K[i, j] = self(a[i], b[j])
        return K

    @staticmethod
    def from_numpy(a: np.ndarray, b: np.ndarray, num_qubits: int = 4) -> np.ndarray:
        """Static helper that evaluates the kernel with a NumPy backend."""
        dev = qml.device("default.qubit", wires=num_qubits)

        @qml.qnode(dev, interface="numpy")
        def circuit(x, y):
            for i in range(num_qubits):
                qml.RY(x[i], wires=i)
            for i in range(num_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            for i in range(num_qubits):
                qml.RY(-y[i], wires=i)
            return np.expval(qml.PauliZ(0))

        n, m = a.shape[0], b.shape[0]
        K = np.empty((n, m))
        for i in range(n):
            for j in range(m):
                K[i, j] = abs(circuit(a[i], b[j]))
        return K

__all__ = ["QuantumKernelMethod"]
