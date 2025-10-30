import pennylane as qml
from pennylane import numpy as np
import torch
from torch import nn
from pennylane.qnn import QNN

class QCNNQuantum(nn.Module):
    """
    PennyLane implementation of the QCNN architecture.
    Feature map: ZFeatureMap over 8 qubits.
    Ansatz: 3 convolution‑pool layers with trainable parameters.
    The quantum circuit is wrapped in a PyTorch QNN for seamless back‑propagation.
    """
    def __init__(self, dev: qml.Device | None = None, seed: int | None = None):
        super().__init__()
        if seed is not None:
            np.random.seed(seed)
            qml.set_options(seed=seed)
        self.n_qubits = 8
        self.dev = dev or qml.device("default.qubit", wires=self.n_qubits)

        self.feature_map = qml.templates.ZFeatureMap(self.n_qubits, reps=1)

        # Layer weights
        self.weights = {
            "c1": {"theta": np.random.uniform(0, 2*np.pi, (4, 3))},
            "p1": {"theta": np.random.uniform(0, 2*np.pi, (2, 3))},
            "c2": {"theta": np.random.uniform(0, 2*np.pi, (2, 3))},
            "p2": {"theta": np.random.uniform(0, 2*np.pi, (1, 3))},
            "c3": {"theta": np.random.uniform(0, 2*np.pi, (1, 3))},
            "p3": {"theta": np.random.uniform(0, 2*np.pi, (1, 3))},
        }

        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs, weights):
            self.feature_map(inputs, wires=range(self.n_qubits))
            self._conv_layer(weights["c1"]["theta"], 8)
            self._pool_layer(weights["p1"]["theta"], 4)
            self._conv_layer(weights["c2"]["theta"], 4)
            self._pool_layer(weights["p2"]["theta"], 2)
            self._conv_layer(weights["c3"]["theta"], 2)
            self._pool_layer(weights["p3"]["theta"], 1)
            return qml.expval(qml.PauliZ(0))

        self.qnn = QNN(circuit, weights=self.weights)

    def _conv_layer(self, theta, n_qubits):
        for i in range(0, n_qubits, 2):
            self._conv_circuit(theta[i//2], i, i+1)

    def _pool_layer(self, theta, n_qubits):
        for i in range(0, n_qubits, 2):
            self._pool_circuit(theta[i//2], i, i+1)

    def _conv_circuit(self, theta, q1, q2):
        qml.RZ(-np.pi/2, wires=q2)
        qml.CNOT(wires=[q2, q1])
        qml.RZ(theta[0], wires=q1)
        qml.RY(theta[1], wires=q2)
        qml.CNOT(wires=[q1, q2])
        qml.RY(theta[2], wires=q2)
        qml.CNOT(wires=[q2, q1])
        qml.RZ(np.pi/2, wires=q1)

    def _pool_circuit(self, theta, q1, q2):
        qml.RZ(-np.pi/2, wires=q2)
        qml.CNOT(wires=[q2, q1])
        qml.RZ(theta[0], wires=q1)
        qml.RY(theta[1], wires=q2)
        qml.CNOT(wires=[q1, q2])
        qml.RY(theta[2], wires=q2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [self.qnn(x[i]) for i in range(x.shape[0])]
        out_tensor = torch.stack(outputs)
        return torch.sigmoid(out_tensor)

def QCNN() -> QCNNQuantum:
    """Factory returning the configured quantum QCNN."""
    return QCNNQuantum()

__all__ = ["QCNN", "QCNNQuantum"]
