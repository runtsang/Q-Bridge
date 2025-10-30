import pennylane as qml
import torch
import numpy as np

class QuantumKernelPennylane(torch.nn.Module):
    """Quantum kernel based on Pennylane using a variational ansatz."""
    def __init__(self, n_qubits: int = 4, depth: int = 2, device: str = 'default.qubit'):
        super().__init__()
        self.n_qubits = n_qubits
        self.depth = depth
        self.device = device
        self.qnode = qml.QNode(self._circuit,
                               qml.device(self.device, wires=self.n_qubits))

    def _circuit(self, x: np.ndarray, y: np.ndarray) -> float:
        # Feature encoding: rotate each qubit by x and y
        for i in range(self.n_qubits):
            qml.RX(x[i], wires=i)
        # Entanglement layer
        for i in range(self.n_qubits - 1):
            qml.CZ(i, i + 1)
        # Encode second vector with negative rotation
        for i in range(self.n_qubits):
            qml.RX(-y[i], wires=i)
        # Entanglement again
        for i in range(self.n_qubits - 1):
            qml.CZ(i, i + 1)
        # Return overlap as expectation value of PauliZ on wire 0
        return qml.expval(qml.PauliZ(0))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Convert to numpy for Pennylane
        x_np = x.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        return torch.tensor(self.qnode(x_np, y_np), dtype=torch.float32, device=x.device)

def quantum_kernel_matrix(a: np.ndarray,
                          b: np.ndarray,
                          n_qubits: int = 4,
                          depth: int = 2,
                          device: str = 'default.qubit') -> np.ndarray:
    kernel = QuantumKernelPennylane(n_qubits, depth, device)
    return np.array([[kernel(torch.tensor(x, dtype=torch.float32),
                            torch.tensor(y, dtype=torch.float32))
                     for y in b] for x in a])

# Optional: quantum LSTM cell using Pennylane (illustrative)
class QLSTMCell(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        # Define a simple variational ansatz for gates
        self.encoder = qml.QNode(self._circuit,
                                 qml.device('default.qubit', wires=self.n_qubits))
        # Linear layers to map classical inputs to quantum parameters
        self.to_params = torch.nn.Linear(self.input_dim + self.hidden_dim, self.n_qubits)

    def _circuit(self, params: np.ndarray) -> np.ndarray:
        for i in range(self.n_qubits):
            qml.RX(params[i], wires=i)
        return qml.expval(qml.PauliZ(0))

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([x, h], dim=-1)
        params = self.to_params(combined).detach().cpu().numpy()
        return torch.tensor(self.encoder(params), dtype=torch.float32, device=x.device)
