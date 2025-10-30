import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F

class QuanvolutionFilter(nn.Module):
    """
    Quantum quanvolution filter that applies a parameterised circuit
    to each 2x2 patch of a 28x28 image. The circuit encodes the patch
    values into qubit rotations, applies a trainable entangling layer,
    and measures Pauli‑Z expectation values.
    """
    def __init__(self, n_wires=4, n_layers=2, device='default.qubit'):
        super().__init__()
        self.n_wires = n_wires
        self.n_layers = n_layers
        self.dev = qml.device(device, wires=self.n_wires)
        # Trainable parameters: shape (n_layers, n_wires, 3) for RY, RZ, RX
        self.params = nn.Parameter(torch.randn(self.n_layers, self.n_wires, 3))
        # Define the quantum node
        self.qnode = qml.QNode(self._circuit, self.dev, interface='torch')

    def _circuit(self, patch, params):
        # Encode pixel intensities into Ry rotations
        for i in range(self.n_wires):
            qml.RY(patch[i], wires=i)
        # Variational layers
        for layer in range(self.n_layers):
            for i in range(self.n_wires):
                qml.RY(params[layer, i, 0], wires=i)
                qml.RZ(params[layer, i, 1], wires=i)
                qml.RX(params[layer, i, 2], wires=i)
            # Entangling layer
            for i in range(self.n_wires - 1):
                qml.CNOT(wires=[i, i+1])
            qml.CNOT(wires=[self.n_wires-1, 0])
        # Measure expectation of Z on all wires
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_wires)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = torch.stack([x[:, r, c],
                                     x[:, r, c+1],
                                     x[:, r+1, c],
                                     x[:, r+1, c+1]], dim=1)
                # Compute expectation values for each sample in the batch
                expvals = torch.stack([self.qnode(patch[i], self.params) for i in range(bsz)], dim=0)
                patches.append(expvals)
        return torch.cat(patches, dim=1)

class QuanvolutionClassifier(nn.Module):
    """
    Hybrid network that uses the quantum quanvolution filter followed by
    a linear head. The filter is end‑to‑end differentiable thanks to
    Pennylane’s autograd support.
    """
    def __init__(self, num_classes=10, n_layers=2, dropout=0.2):
        super().__init__()
        self.qfilter = QuanvolutionFilter(n_layers=n_layers)
        self.linear = nn.Linear(4 * 14 * 14, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        logits = self.dropout(logits)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionFilter", "QuanvolutionClassifier"]
