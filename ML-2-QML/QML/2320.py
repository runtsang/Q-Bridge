"""Hybrid quantum model combining quanvolution filter with a variational regression head.

The model can perform classification or regression on image data.  It uses a
quantum 2×2 patch encoder (QuanvolutionFilter) followed by a variational
circuit that maps the encoded features to a regression output.  The
classification mode uses a linear head on the quantum measurement results.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchquantum as tq


class QuanvolutionFilter(tq.QuantumModule):
    """Apply a random two‑qubit quantum kernel to 2×2 image patches."""
    def __init__(self) -> None:
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


class HybridQuanvolutionModel(tq.QuantumModule):
    """Hybrid quanvolution model with optional regression head."""
    class _RegressionLayer(tq.QuantumModule):
        def __init__(self, num_wires: int) -> None:
            super().__init__()
            self.n_wires = num_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random_layer(qdev)
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)

    def __init__(self, num_wires: int = 4, regression: bool = False) -> None:
        super().__init__()
        self.num_wires = num_wires
        self.filter = QuanvolutionFilter()
        # Reduce the 4×14×14 feature vector to num_wires for encoding
        self.feature_reducer = nn.Linear(4 * 14 * 14, num_wires)
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self._RegressionLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.regression = regression
        if regression:
            self.head = nn.Linear(num_wires, 1)
        else:
            self.classifier = nn.Linear(num_wires, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        # Quantum filter produces 4*14*14 features
        raw_features = self.filter(x)  # shape [bsz, 784]
        # Classical reduction to num_wires
        reduced = self.feature_reducer(raw_features)
        # Encode reduced features into quantum state
        qdev = tq.QuantumDevice(n_wires=self.num_wires, bsz=bsz, device=device)
        self.encoder(qdev, reduced)
        self.q_layer(qdev)
        out = self.measure(qdev)
        if self.regression:
            return self.head(out).squeeze(-1)
        else:
            logits = self.classifier(out)
            return torch.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionFilter", "HybridQuanvolutionModel"]
