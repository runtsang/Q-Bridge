"""Hybrid quanvolution model – quantum implementation.

The quantum filter replaces the classical conv with a 2×2 patch encoder
followed by a random circuit and a small trainable gate block.  The
classifier then maps the resulting 4×14×14 feature map through a
quantum fully‑connected block and a linear read‑out to produce the
10‑class log‑softmax output.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class HybridQuanvolutionFilter(tq.QuantumModule):
    """Quantum filter that processes 2×2 image patches into 4‑qubit
    measurement outcomes.

    The filter uses a general encoder that maps each pixel to a
    Ry rotation on a dedicated qubit, then applies a random layer
    followed by a small trainable gate sequence.  The measurement
    of all qubits yields a 4‑dimensional feature vector per patch.
    """
    def __init__(
        self,
        kernel_size: int = 2,
        n_wires: int = 4,
        random_ops: int = 8,
    ) -> None:
        super().__init__()
        self.n_wires = n_wires
        # Encode each pixel as a Y‑rotation on a separate qubit
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.random_layer = tq.RandomLayer(n_ops=random_ops, wires=list(range(n_wires)))
        # Small trainable gate block
        self.rx0 = tq.RX(has_params=True, trainable=True)
        self.ry0 = tq.RY(has_params=True, trainable=True)
        self.rz0 = tq.RZ(has_params=True, trainable=True)
        self.crx0 = tq.CRX(has_params=True, trainable=True)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=x.device)
        # 28×28 image patches of size 2×2 → 14×14 grid
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
                )
                self.encoder(qdev, patch)
                self.random_layer(qdev)
                self.rx0(qdev, wires=0)
                self.ry0(qdev, wires=1)
                self.rz0(qdev, wires=3)
                self.crx0(qdev, wires=[0, 2])
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        # Concatenate all patch measurements → (bsz, 4*14*14)
        return torch.cat(patches, dim=1)


class HybridQuanvolutionClassifier(tq.QuantumModule):
    """Quantum classifier that follows the hybrid filter with a
    quantum‑fully‑connected block and a linear read‑out.
    """
    class _QFCBlock(tq.QuantumModule):
        """Small quantum fully‑connected block inspired by Quantum‑NAT."""
        def __init__(self):
            super().__init__()
            self.n_wires = 4
            self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
            self.random_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
            self.rx0 = tq.RX(has_params=True, trainable=True)
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.rz0 = tq.RZ(has_params=True, trainable=True)
            self.crx0 = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random_layer(qdev)
            self.rx0(qdev, wires=0)
            self.ry0(qdev, wires=1)
            self.rz0(qdev, wires=3)
            self.crx0(qdev, wires=[0, 2])
            tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
            tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
            tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)

    def __init__(
        self,
        num_classes: int = 10,
        dropout_prob: float = 0.0,
    ) -> None:
        super().__init__()
        self.filter = HybridQuanvolutionFilter()
        self.qfc_block = self._QFCBlock()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.linear = nn.Linear(4, num_classes)  # read‑out
        self.batch_norm = nn.BatchNorm1d(num_classes)
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Filter stage: 4×14×14 feature vector
        features = self.filter(x)  # (bsz, 4*14*14)
        bsz = features.shape[0]
        # Reduce to four values per sample for the QFC block
        # Use average pooling to obtain a 4‑dimensional vector
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)  # 16 = 4*4
        qdev = tq.QuantumDevice(n_wires=4, bsz=bsz, device=x.device, record_op=True)
        self.qfc_block.encoder(qdev, pooled)
        self.qfc_block(qdev)
        out = self.measure(qdev)  # (bsz, 4)
        out = self.linear(out)  # (bsz, num_classes)
        out = self.batch_norm(out)
        out = self.dropout(out)
        return F.log_softmax(out, dim=-1)


__all__ = ["HybridQuanvolutionFilter", "HybridQuanvolutionClassifier"]
