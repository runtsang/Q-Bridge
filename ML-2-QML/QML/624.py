"""Quantum-inspired quanvolution with learnable parameters and amplitude‑damping noise."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq

class QuantumFilter(tq.QuantumModule):
    """
    Parameterized two‑qubit circuit with learnable Ry and Rz angles.
    Includes a simple amplitude‑damping noise model.
    """
    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        # 8 learnable parameters per patch (2 per qubit)
        self.params = nn.Parameter(torch.randn(n_wires * 2))
        self.noise = tq.NoiseLayer(tq.NoiseType.AMP_DAMPING, prob=0.02)

    def forward(self, qdev: tq.QuantumDevice, data: torch.Tensor) -> None:
        # Encode data into Ry rotations
        for i in range(self.n_wires):
            qdev.ry(data[:, i], wires=i)
        # Apply learnable rotations
        for i in range(self.n_wires):
            angle_rz = self.params[2 * i]
            angle_ry = self.params[2 * i + 1]
            qdev.rz(angle_rz, wires=i)
            qdev.ry(angle_ry, wires=i)
        # Entangling layer
        qdev.cnot(0, 1)
        qdev.cnot(2, 3)
        # Add noise
        self.noise(qdev)

class QuanvolutionHybrid(nn.Module):
    """
    Hybrid quantum model that replaces the classical 2×2 convolution with
    a learnable quantum kernel. Each 2×2 patch is mapped to 4 qubits,
    processed by QuantumFilter, and measured in Z.
    """
    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.quantum_layer = QuantumFilter(n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
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
                qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
                self.encoder(qdev, data)
                self.quantum_layer(qdev, data)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        features = torch.cat(patches, dim=1)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionHybrid"]
