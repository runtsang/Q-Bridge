"""Quantum counterpart of Quanvolution__gen185: 2×2 patch kernel + quantum fully‑connected head."""
from __future__ import annotations

import torch
import torch.nn as nn
import torchquantum as tq
from typing import List


class QuantumFullyConnected(tq.QuantumModule):
    """Simple quantum fully‑connected layer based on a parameterised Ry circuit."""
    def __init__(self, n_qubits: int = 4):
        super().__init__()
        self.n_qubits = n_qubits
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(n_qubits)]
        )
        self.q_layer = tq.RandomLayer(n_ops=4, wires=range(n_qubits))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_qubits)
        qdev = tq.QuantumDevice(self.n_qubits, bsz=x.shape[0], device=x.device)
        self.encoder(qdev, x)
        self.q_layer(qdev)
        return self.measure(qdev)


class Quanvolution__gen185(tq.QuantumModule):
    """Quantum version: 2×2 patch kernel + quantum fully‑connected head + classical classifier."""
    def __init__(self, patch_qubits: int = 4, fc_qubits: int = 4, num_classes: int = 10):
        super().__init__()
        self.patch_qubits = patch_qubits
        self.fc_qubits = fc_qubits
        self.patch_encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(patch_qubits)]
        )
        self.patch_layer = tq.RandomLayer(n_ops=8, wires=range(patch_qubits))
        self.patch_measure = tq.MeasureAll(tq.PauliZ)
        self.fc = QuantumFullyConnected(fc_qubits)
        self.classifier = nn.Linear(fc_qubits * (28 * 28 // 2 // 2), num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, 28, 28)
        bsz = x.size(0)
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )  # (batch, 4)
                qdev = tq.QuantumDevice(self.patch_qubits, bsz=bsz, device=patch.device)
                self.patch_encoder(qdev, patch)
                self.patch_layer(qdev)
                patches.append(self.patch_measure(qdev).view(bsz, 4))
        feature_map = torch.cat(patches, dim=1)  # (batch, 784)
        # Reshape into blocks for the quantum fully‑connected layer
        num_blocks = feature_map.size(1) // self.fc_qubits
        blocks = feature_map.view(bsz, num_blocks, self.fc_qubits)
        outputs = []
        for block in blocks:
            outputs.append(self.fc(block))
        # Concatenate quantum outputs and classify
        q_outputs = torch.cat(outputs, dim=1)  # (batch, num_blocks * fc_qubits)
        logits = self.classifier(q_outputs)
        return logits

__all__ = ["QuantumFullyConnected", "Quanvolution__gen185"]
