"""Variational quanvolution filter using torchquantum.

This implementation replaces the fixed random layer with a trainable
variational circuit. A learnable linear encoder maps pixel intensities
to rotation angles, and the circuit consists of alternating layers of
parameterized single‑qubit rotations and entangling CNOTs. The module
supports batched execution on a quantum device and is compatible
with PyTorch training loops.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq


class VariationalEncoder(tq.QuantumModule):
    """Learnable encoder that maps 4 pixel values to rotation angles."""
    def __init__(self) -> None:
        super().__init__()
        # Linear layer mapping 4 inputs to 4 rotation angles
        self.linear = nn.Linear(4, 4, bias=False)

    def forward(self, qdev: tq.QuantumDevice, data: torch.Tensor) -> None:
        # data shape: (batch, 4)
        angles = self.linear(data)
        # Apply Ry rotations with the computed angles
        for i in range(4):
            qdev.ry(angles[:, i], wires=i)


class VariationalCircuit(tq.QuantumModule):
    """Parameterized 4‑qubit circuit with entangling layers."""
    def __init__(self, n_layers: int = 2) -> None:
        super().__init__()
        self.n_layers = n_layers
        # Parameters for single‑qubit rotations
        self.rz_params = nn.Parameter(torch.randn(n_layers, 4))
        self.rx_params = nn.Parameter(torch.randn(n_layers, 4))
        # Entanglement pattern: ring of CNOTs
        self.cnot_pattern = [(i, (i + 1) % 4) for i in range(4)]

    def forward(self, qdev: tq.QuantumDevice) -> None:
        for l in range(self.n_layers):
            # Rotation layers
            for i in range(4):
                qdev.rz(self.rz_params[l, i], wires=i)
                qdev.rx(self.rx_params[l, i], wires=i)
            # Entangling layer
            for control, target in self.cnot_pattern:
                qdev.cnot(control, target)


class QuanvolutionFilter(tq.QuantumModule):
    """Variational quanvolution filter applied to 2×2 patches."""
    def __init__(self, n_layers: int = 2) -> None:
        super().__init__()
        self.n_wires = 4
        self.encoder = VariationalEncoder()
        self.circuit = VariationalCircuit(n_layers=n_layers)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 1, 28, 28)
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                # Extract 2x2 patch and flatten
                patch = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                # Encode pixel values into rotation angles
                self.encoder(qdev, patch)
                # Apply trainable circuit
                self.circuit(qdev)
                # Measurement
                meas = self.measure(qdev)
                patches.append(meas.view(bsz, 4))
        return torch.cat(patches, dim=1)


class QuanvolutionClassifier(nn.Module):
    """Hybrid neural network using the variational quanvolution filter."""
    def __init__(self, n_layers: int = 2) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter(n_layers=n_layers)
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionFilter", "QuanvolutionClassifier"]
