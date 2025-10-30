import pennylane as qml
import torch
import numpy as np

dev = qml.device("default.qubit", wires=8)

def conv_layer(params: np.ndarray, wires: list[int]) -> None:
    qml.RZ(-np.pi/2, wires[1])
    qml.CNOT(wires[1], wires[0])
    qml.RZ(params[0], wires[0])
    qml.RY(params[1], wires[1])
    qml.CNOT(wires[0], wires[1])
    qml.RY(params[2], wires[1])
    qml.CNOT(wires[1], wires[0])
    qml.RZ(np.pi/2, wires[0])

def pool_layer(params: np.ndarray, wires: list[int]) -> None:
    qml.RZ(-np.pi/2, wires[1])
    qml.CNOT(wires[1], wires[0])
    qml.RZ(params[0], wires[0])
    qml.RY(params[1], wires[1])
    qml.CNOT(wires[0], wires[1])
    qml.RY(params[2], wires[1])

def feature_map(x: torch.Tensor, wires: list[int]) -> None:
    for i, val in enumerate(x):
        qml.RY(val, wires[i])

@qml.qnode(dev, interface="torch")
def qcnn_model(x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    feature_map(x, range(8))
    # Convolutional layers
    conv_layer(params[0:3], [0, 1])
    conv_layer(params[3:6], [2, 3])
    conv_layer(params[6:9], [4, 5])
    conv_layer(params[9:12], [6, 7])
    # Pooling layers
    pool_layer(params[12:15], [0, 1])
    pool_layer(params[15:18], [2, 3])
    pool_layer(params[18:21], [4, 5])
    pool_layer(params[21:24], [6, 7])
    # Final observable
    return qml.expval(qml.PauliZ(0))

class QCNNGen164:
    """Hybrid QCNN implemented with Pennylane."""
    def __init__(self, n_params: int = 24) -> None:
        self.n_params = n_params
        self.params = torch.randn(n_params, requires_grad=True)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return qcnn_model(x, self.params)

def QCNNGen164() -> QCNNGen164:
    """Factory returning a configured QCNNGen164 instance."""
    return QCNNGen164()

__all__ = ["QCNNGen164", "QCNNGen164"]
