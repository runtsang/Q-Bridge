import torch
from torch import nn
import pennylane as qml
import pennylane.numpy as np

class QCNNModel(nn.Module):
    """
    Hybrid quantum‑classical QCNN implemented with PennyLane.
    Uses a Z‑feature map, a depth‑3 ansatz with tunable rotation angles,
    and full‑chain CNOT entanglement.  The model is differentiable via
    the TorchLayer wrapper.
    """

    def __init__(self,
                 n_qubits: int = 8,
                 device: str = 'default.qubit',
                 seed: int = 12345) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.device = qml.device(device, wires=n_qubits)
        np.random.seed(seed)

        # Feature map – simple Z‑feature map
        def feature_map(x):
            for i in range(n_qubits):
                qml.RZ(x[i], i)

        # Ansatz – 3 layers of Ry rotations + full‑chain entanglement
        def ansatz(params):
            for layer in range(3):
                for i in range(n_qubits):
                    qml.RY(params[layer, i], i)
                # Full‑chain entanglement
                for i in range(n_qubits - 1):
                    qml.CNOT(i, i + 1)
                qml.CNOT(n_qubits - 1, 0)

        # QNode combining feature map and ansatz
        def circuit(x, weights):
            feature_map(x)
            ansatz(weights)
            return qml.expval(qml.PauliZ(0))

        self.qnode = qml.QNode(circuit, self.device, interface='torch')
        # Initialise weights: 3 layers × n_qubits parameters
        init = torch.randn(3, n_qubits, dtype=torch.float64)
        self.weight_params = nn.Parameter(init)
        # TorchLayer for automatic differentiation
        self.layer = qml.qnn.TorchLayer(self.qnode, weights=self.weight_params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Rescale expectation to [0, 1] with sigmoid
        return torch.sigmoid(self.layer(x))

def QCNN() -> QCNNModel:
    """
    Factory returning a ready‑to‑train :class:`QCNNModel` using PennyLane.
    """
    return QCNNModel()

__all__ = ["QCNN", "QCNNModel"]
