import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import numpy as np

class QuantumSelfAttention(tq.QuantumModule):
    """Variational self‑attention implemented with a small quantum circuit."""
    def __init__(self, n_qubits: int = 4):
        super().__init__()
        self.n_qubits = n_qubits
        # encode each input value as a rotation around Y
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        # variational layer that mixes the qubits
        self.var_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_qubits)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_qubits)
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(self.n_qubits, bsz=bsz, device=x.device)
        self.encoder(qdev, x)
        self.var_layer(qdev)
        return self.measure(qdev).view(bsz, -1)

class QuanvolutionFilterQuantum(tq.QuantumModule):
    """Quantum analogue of a 2×2 patchwise filter."""
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, 28, 28)
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
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)

class HybridQuanvolutionAttentionClassifier(tq.QuantumModule):
    """Fully quantum pipeline mirroring the classical hybrid."""
    def __init__(self, num_classes: int = 10, embed_dim: int = 4):
        super().__init__()
        self.qfilter = QuanvolutionFilterQuantum()
        self.attention = QuantumSelfAttention(n_qubits=embed_dim)
        self.linear = nn.Linear(embed_dim * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Phase 1: quantum quanvolution
        qfeat = self.qfilter(x)  # (batch, 4*14*14)
        seq_len = qfeat.shape[1] // self.attention.n_qubits
        qfeat_seq = qfeat.view(-1, seq_len, self.attention.n_qubits)
        # Phase 2: apply quantum self‑attention to each patch
        attn_features = []
        for i in range(seq_len):
            slice = qfeat_seq[:, i, :]
            attn = self.attention(slice)
            attn_features.append(attn)
        attn_features = torch.cat(attn_features, dim=1)  # (batch, seq_len*embed_dim)
        # Phase 3: classification head
        logits = self.linear(attn_features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["HybridQuanvolutionAttentionClassifier"]
