# QML code
"""Quantum-enhanced quanvolution filter using a parameterized variational circuit per patch."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq

class VariationalQuantumLayer(tq.QuantumModule):
    'Parameterized variational circuit with two qubits per patch.'
    def __init__(self, n_wires: int = 4, depth: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.depth = depth
        # Parameterized rotation gates
        self.params = nn.Parameter(torch.randn(depth, n_wires, 3))
        # Entangling layer
        # No explicit entangling layer; use CNOT inside forward

    def forward(self, qdev: tq.QuantumDevice, data: torch.Tensor) -> tq.QuantumDevice:
        # data shape: (batch, n_wires)
        for i in range(self.depth):
            for w in range(self.n_wires):
                # Apply Ry, Rz, Rx with trainable parameters
                tq.Ry(self.params[i, w, 0])(qdev, wires=[w])
                tq.Rz(self.params[i, w, 1])(qdev, wires=[w])
                tq.Rx(self.params[i, w, 2])(qdev, wires=[w])
            # Entangle adjacent qubits
            for w in range(self.n_wires - 1):
                tq.CNOT(qdev, wires=[w, w + 1])
        return qdev

class QuanvolutionFilter(tq.QuantumModule):
    'Apply a variational quantum kernel to 2x2 image patches.'
    def __init__(self, n_wires: int = 4, depth: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [
                {'input_idx': [0], 'func': 'ry', 'wires': [0]},
                {'input_idx': [1], 'func': 'ry', 'wires': [1]},
                {'input_idx': [2], 'func': 'ry', 'wires': [2]},
                {'input_idx': [3], 'func': 'ry', 'wires': [3]},
            ]
        )
        self.var_layer = VariationalQuantumLayer(n_wires=n_wires, depth=depth)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                data = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, data)
                self.var_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)

class QuanvolutionClassifier(nn.Module):
    'Hybrid neural network using the quantum quanvolution filter followed by a classical head.'
    def __init__(self, num_classes: int = 10, n_wires: int = 4, depth: int = 4):
        super().__init__()
        self.qfilter = QuanvolutionFilter(n_wires=n_wires, depth=depth)
        self.linear = nn.Linear(4 * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ['VariationalQuantumLayer', 'QuanvolutionFilter', 'QuanvolutionClassifier']
