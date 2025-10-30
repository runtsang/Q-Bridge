"""HybridQuantumNAT: Quantum‑enhanced variant using torchquantum."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class HybridQuantumNAT(tq.QuantumModule):
    """
    Hybrid architecture where the final regression head is replaced by a variational quantum circuit.

    Architecture:
        - Classical CNN feature extractor identical to the pure ML version.
        - 1‑D average pooling of the feature map to produce 16‑dim vector.
        - General 4‑qubit encoder (ryzxy) maps the 16‑dim vector into a 4‑qubit state.
        - Quantum variational layer: random unitary + trainable RX, RY, RZ, CRX gates.
        - Measurement of Pauli‑Z on all qubits.
        - Batch‑norm + linear projection to scalar output.
    """

    class QuantumLayer(tq.QuantumModule):
        def __init__(self) -> None:
            super().__init__()
            self.n_wires = 4
            # Random layer for entanglement and expressibility
            self.random_layer = tq.RandomLayer(
                n_ops=50, wires=list(range(self.n_wires))
            )
            # Trainable single‑qubit rotations
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)
            self.crx = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random_layer(qdev)
            self.rx(qdev, wires=0)
            self.ry(qdev, wires=1)
            self.rz(qdev, wires=3)
            self.crx(qdev, wires=[0, 2])
            # Add a few deterministic gates for structure
            tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
            tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
            tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)

    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        # Classical feature extractor (identical to the ML version)
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # Encoder that maps 16‑dim feature vector to 4 qubits
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict["4x4_ryzxy"]
        )
        self.q_layer = self.QuantumLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        # Classical feature extraction
        feats = self.features(x)
        # 1D average pooling per channel to obtain 16‑dim vector
        pooled = F.avg_pool2d(feats, kernel_size=6).view(bsz, 16)
        # Quantum device
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True
        )
        # Encode classical data into quantum state
        self.encoder(qdev, pooled)
        # Apply variational quantum circuit
        self.q_layer(qdev)
        # Measure all qubits
        out = self.measure(qdev)
        # Simple linear post‑processing
        out = self.norm(out)
        # Optional small classical head to collapse to scalar
        out = torch.mean(out, dim=1, keepdim=True)
        return out


__all__ = ["HybridQuantumNAT"]
