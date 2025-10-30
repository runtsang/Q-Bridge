import torch
import torch.nn as nn
import torchquantum as tq
import numpy as np

class QuantumQuanvolutionFilter(tq.QuantumModule):
    """Quantum version of the quanvolution filter using Ry encoding and random layers."""
    def __init__(self, n_wires=4, n_ops=8):
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
        self.random_layer = tq.RandomLayer(n_ops=n_ops, wires=list(range(n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x):
        bsz = x.shape[0]
        x = x.view(bsz, 28, 28)
        patches = []
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=x.device)
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
                self.random_layer(qdev)
                meas = self.measure(qdev)
                patches.append(meas.view(bsz, 4))
        flat = torch.cat(patches, dim=1)
        flat = tq.BatchNorm1d(4 * 14 * 14)(flat)
        return flat

class QuantumSelfAttention(tq.QuantumModule):
    """Quantum self‑attention block using parameterised rotations and entanglement."""
    def __init__(self, n_qubits=4):
        super().__init__()
        self.n_qubits = n_qubits
        self.rotation = tq.RandomLayer(n_ops=3 * n_qubits, wires=list(range(n_qubits)))
        self.entangle = tq.RandomLayer(n_ops=n_qubits - 1, wires=list(range(n_qubits - 1)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x):
        bsz, _ = x.shape
        qdev = tq.QuantumDevice(self.n_qubits, bsz=bsz, device=x.device)
        self.rotation(qdev, x)
        self.entangle(qdev)
        meas = self.measure(qdev)
        probs = torch.softmax(meas, dim=-1)
        return probs @ x

class QuanvolutionAttentionModel(tq.QuantumModule):
    """End‑to‑end quantum‑classical hybrid mirroring the classical architecture."""
    def __init__(self, num_classes=10, n_qubits=4):
        super().__init__()
        self.filter = QuantumQuanvolutionFilter()
        self.attention = QuantumSelfAttention(n_qubits=n_qubits)
        self.head = nn.Linear(4 * 14 * 14, num_classes)

    def forward(self, x):
        x = self.filter(x)
        x = self.attention(x)
        logits = self.head(x)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionAttentionModel"]
