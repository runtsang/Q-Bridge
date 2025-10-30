"""
Hybrid quantum model that mirrors the classical architecture:
    - Quantum encoder processes 2×2 image patches into a 4‑wire circuit.
    - A variational QFC block with random and parameterised gates.
    - Measurement of all qubits followed by batch‑normalisation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf


class QuantumQuanvolutionFilter(tq.QuantumModule):
    """
    Quantum analogue of the classical quanvolution filter.
    Each 2×2 patch is encoded into a 4‑wire circuit using Ry gates,
    then passed through a random layer before measurement.
    """
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        # Encode each pixel of the patch as an Ry rotation
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.random_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        # Reshape to (batch, 28, 28) for MNIST‑style images
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
                self.random_layer(qdev)
                patches.append(self.measure(qdev).view(bsz, 4))
        return torch.cat(patches, dim=1)


class QuantumFullyConnected(tq.QuantumModule):
    """
    Variational block that emulates the fully‑connected projection of
    the classical head.  It uses a random layer followed by
    parameterised RX/RZ gates and a controlled‑RX, matching the QFCModel
    from the original Quantum‑NAT seed.
    """
    class QLayer(tq.QuantumModule):
        def __init__(self) -> None:
            super().__init__()
            self.n_wires = 4
            self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
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

    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=x.device, record_op=True)
        # Average pool the input to match the 4‑wire encoding size
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        # Encode the pooled features into the quantum circuit
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)


class HybridQuantumNAT(tq.QuantumModule):
    """
    End‑to‑end quantum model that combines the quanvolutional filter
    with a variational fully‑connected block.  The architecture is
    purposely symmetrical to the classical HybridQuantumNAT.
    """
    def __init__(self) -> None:
        super().__init__()
        self.qfilter = QuantumQuanvolutionFilter()
        self.qfc = QuantumFullyConnected()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.qfilter(x)
        out = self.qfc(features)
        return out


__all__ = ["HybridQuantumNAT"]
