import pennylane as qml
import pennylane.numpy as np
import torch
from torch import nn

# Quantum device
dev = qml.device("default.qubit", wires=8)

def conv_layer(params, wires):
    """Two‑qubit convolution block parameterised by params of shape (pairs, 3)."""
    for i, (q1, q2) in enumerate(zip(wires[0::2], wires[1::2])):
        p = params[i]
        qml.RZ(-np.pi/2, wires=q2)
        qml.CNOT(q2, q1)
        qml.RZ(p[0], wires=q1)
        qml.RY(p[1], wires=q2)
        qml.CNOT(q1, q2)
        qml.RY(p[2], wires=q2)
        qml.CNOT(q2, q1)
        qml.RZ(np.pi/2, wires=q1)

def pool_layer(params, wires):
    """Two‑qubit pooling block parameterised by params of shape (pairs, 3)."""
    for i, (s, t) in enumerate(zip(wires[0::2], wires[1::2])):
        p = params[i]
        qml.RZ(-np.pi/2, wires=t)
        qml.CNOT(t, s)
        qml.RZ(p[0], wires=s)
        qml.RY(p[1], wires=t)
        qml.CNOT(s, t)
        qml.RY(p[2], wires=t)

def feature_map(inputs):
    """Z‑feature map used in the original QCNN."""
    for i, val in enumerate(inputs):
        qml.RZ(val, wires=i)

@qml.qnode(dev, interface="torch")
def qcircuit(inputs, weights):
    feature_map(inputs)
    # Layer 1
    conv_layer(weights[0], range(8))
    pool_layer(weights[1], range(8))
    # Layer 2
    conv_layer(weights[2], range(4, 8))
    pool_layer(weights[3], range(4, 8))
    # Layer 3
    conv_layer(weights[4], range(6, 8))
    pool_layer(weights[5], range(6, 8))
    return qml.expval(qml.PauliZ(0))

class QCNNModel(nn.Module):
    """Hybrid QCNN: a quantum layer followed by a tiny classical head."""
    def __init__(self):
        super().__init__()
        # Total parameters: 12 + 12 + 6 + 6 + 3 + 3 = 42
        self.weight_vector = nn.Parameter(torch.randn(42))
        self.classifier = nn.Linear(1, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs shape: (batch, 8)
        batch_size = inputs.shape[0]
        # Slice weight vector into per‑layer blocks
        sizes = [12, 12, 6, 6, 3, 3]
        weights = []
        idx = 0
        for s in sizes:
            layer = self.weight_vector[idx:idx + s].reshape(-1, 3)
            weights.append(layer)
            idx += s
        # Evaluate quantum circuit for each sample
        out = torch.stack([qcircuit(inp, weights) for inp in inputs])
        out = self.classifier(out)
        return torch.sigmoid(out)

def QCNN() -> QCNNModel:
    """Factory returning the configured QCNNModel."""
    return QCNNModel()

__all__ = ["QCNN", "QCNNModel"]
