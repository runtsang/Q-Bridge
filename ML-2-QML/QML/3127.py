"""Hybrid quantum model inspired by Quanvolution and Quantum‑NAT."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class QuantumQuanvolutionFilter(tq.QuantumModule):
    """
    Apply a tunable RandomLayer to each 2×2 patch of a 28×28 image.
    The encoder uses a 4‑wire general Ry encoder, followed by a RandomLayer
    and a measurement of all qubits. The result is a 4‑dimensional feature
    per patch, concatenated into a vector of length 4×14×14.
    """
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
        self.q_layer = tq.RandomLayer(n_ops=12, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    @tq.static_support
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=device)
        x = x.view(bsz, 28, 28)  # assume grayscale MNIST
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
                self.q_layer(qdev)
                meas = self.measure(qdev)
                patches.append(meas.view(bsz, 4))
        return torch.cat(patches, dim=1)  # shape: (bsz, 4*14*14)


class QuantumFullyConnectedLayer(tq.QuantumModule):
    """
    Quantum fully‑connected layer inspired by the QFCModel's QLayer.
    Uses a RandomLayer followed by trainable RX, RY, RZ, CRX gates and
    simple Hadamard/SX/CNOT operations to mix amplitudes before measurement.
    """
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.random = tq.RandomLayer(n_ops=30, wires=list(range(self.n_wires)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)
        self.rz = tq.RZ(has_params=True, trainable=True)
        self.crx = tq.CRX(has_params=True, trainable=True)

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
        self.random(qdev)
        self.rx(qdev, wires=0)
        self.ry(qdev, wires=1)
        self.rz(qdev, wires=3)
        self.crx(qdev, wires=[0, 2])
        tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
        tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
        tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)
        return tq.MeasureAll(tq.PauliZ)(qdev)  # measurement


class QuantumHybridClassifier(tq.QuantumModule):
    """
    End‑to‑end quantum model: patch encoder → quantum fully‑connected layer → log‑softmax.
    The encoder produces a 4‑dimensional feature per patch; the fully‑connected layer
    acts on the flattened vector of 4×14×14 qubits, yielding 10 logits.
    """
    def __init__(self) -> None:
        super().__init__()
        self.filter = QuantumQuanvolutionFilter()
        self.qfc = QuantumFullyConnectedLayer()
        self.norm = nn.BatchNorm1d(10)

    @tq.static_support
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.filter.n_wires, bsz=bsz, device=x.device, record_op=True)
        # Encode image patches
        patch_features = self.filter(x)  # shape: (bsz, 4*14*14)
        # Reshape to match quantum device wires
        qdev.set_state(patch_features)  # assume state preparation via measurement results
        # Apply quantum fully‑connected layer
        out = self.qfc(qdev)  # shape: (bsz, 4)
        out = self.norm(out)
        return F.log_softmax(out, dim=-1)


__all__ = ["QuantumQuanvolutionFilter", "QuantumFullyConnectedLayer", "QuantumHybridClassifier"]
