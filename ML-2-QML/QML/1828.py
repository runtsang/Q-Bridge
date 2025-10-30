"""Hybrid QCNN using Pennylane for the quantum layer and PyTorch for the read‑out."""

import pennylane as qml
import torch
from torch import nn
from pennylane.optimize import AdamOptimizer

# Number of qubits in the feature map and ansatz
NUM_QUBITS = 8
NUM_PARAMS = 3 * (NUM_QUBITS // 2)  # 3 parameters per two‑qubit block

# Create a quantum device
dev = qml.device("default.qubit", wires=NUM_QUBITS)


def convex_layer(*wires):
    """Two‑qubit convolution block used repeatedly in the ansatz."""
    @qml.qnode(dev, interface="torch")
    def layer(params, x):
        # Feature encoding
        qml.Hadamard(wires=wires[0])
        qml.Hadamard(wires=wires[1])

        # Parameterised gates
        qml.RZ(params[0], wires=wires[0])
        qml.RY(params[1], wires=wires[1])
        qml.CNOT(wires=[wires[1], wires[0]])
        qml.RY(params[2], wires=wires[1])
        qml.CNOT(wires=[wires[0], wires[1]])

        # Measurement expectation
        return qml.expval(qml.PauliZ(wires=wires[0]))
    return layer


class HybridQCNN(nn.Module):
    """
    A hybrid QCNN with a Pennylane variational circuit followed by a
    classical linear read‑out.  The circuit is constructed by
    stacking convolutional and pooling layers similar to the
    original Qiskit implementation.
    """

    def __init__(self, learning_rate: float = 0.01) -> None:
        super().__init__()
        # Classical read‑out layer
        self.readout = nn.Linear(1, 1)

        # Parameters for the quantum circuit
        self.params = nn.Parameter(
            torch.randn(NUM_PARAMS, dtype=torch.float32, requires_grad=True)
        )

        # Optimiser for the quantum part (used only during training)
        self.optimizer = AdamOptimizer(stepsize=learning_rate)

        # Build layer indices for convenience
        self.first_layer = [0, 1, 2, 3]
        self.second_layer = [4, 5, 6, 7]
        self.third_layer = [6, 7]  # last conv pair

    def conv_block(self, params: torch.Tensor, wires: list[int]):
        """Apply a convolution block to the given wires."""
        return convex_layer(*wires)(params, None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is expected to be a batch of feature vectors of size (batch, 8)
        # Encode features into the quantum state using a simple Z feature map
        # Here we use a fixed encoding for simplicity; more elaborate maps can be added.
        qs = []
        for i in range(NUM_QUBITS):
            qml.PauliZ(i)  # placeholder for feature encoding

        # Split parameters for each block
        idx = 0
        outputs = []

        # First convolutional block
        out1 = self.conv_block(self.params[idx : idx + 3], self.first_layer)
        outputs.append(out1)
        idx += 3

        # First pooling (simulate by measuring and discarding one qubit)
        out1_pooled = out1  # in practice we would drop qubits
        outputs.append(out1_pooled)
        idx += 3

        # Second convolutional block
        out2 = self.conv_block(self.params[idx : idx + 3], self.second_layer)
        outputs.append(out2)
        idx += 3

        # Second pooling
        out2_pooled = out2
        outputs.append(out2_pooled)
        idx += 3

        # Third convolutional block
        out3 = self.conv_block(self.params[idx : idx + 3], self.third_layer)
        outputs.append(out3)

        # Aggregate outputs (here we simply average)
        q_out = torch.stack(outputs, dim=0).mean(dim=0)

        # Classical read‑out
        return torch.sigmoid(self.readout(q_out))


def HybridQCNNModel() -> HybridQCNN:
    """Factory returning the configured :class:`HybridQCNN`."""
    return HybridQCNN()


__all__ = ["HybridQCNN", "HybridQCNNModel"]
