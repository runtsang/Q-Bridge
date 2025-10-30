"""Quantum‑enhanced variant of the hybrid CNN.

The quantum module shares the same interface as the classical
``QFCHybridModel`` but replaces the fully‑connected head with a
parameterised variational circuit that acts on a 16‑qubit state
prepared from the classical feature vector.  The encoder uses the
``4x4_ryzxy`` template and the variational block combines random
operations with trainable single‑qubit rotations and a controlled‑RX.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class QFCHybridModel(tq.QuantumModule):
    """Quantum‑enhanced CNN that produces a 4‑dimensional output.

    The architecture mirrors the classical counterpart but the final
    classification head is a variational quantum circuit.  The
    ``encode_to_quantum_vector`` method is inherited from the
    classical model and is used to prepare the quantum state.
    """

    class QLayer(tq.QuantumModule):
        """Variational block applied after the classical encoder."""

        def __init__(self) -> None:
            super().__init__()
            self.n_wires = 16
            # Random layer with 120 operations
            self.random_layer = tq.RandomLayer(n_ops=120, wires=list(range(self.n_wires)))
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
            self.rz(qdev, wires=2)
            self.crx(qdev, wires=[3, 4])
            tqf.hadamard(qdev, wires=5, static=self.static_mode, parent_graph=self.graph)
            tqf.sx(qdev, wires=6, static=self.static_mode, parent_graph=self.graph)
            tqf.cnot(qdev, wires=[7, 8], static=self.static_mode, parent_graph=self.graph)

    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 16
        # Classical backbone shared with the ML version
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.res_block = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
        )

        self.dropout = nn.Dropout2d(p=0.25)

        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        # Classical feature extraction
        x = self.features(x)
        residual = self.res_block(x)
        x = F.relu(x + residual)
        x = self.dropout(x)
        # Prepare quantum state
        pooled = F.avg_pool2d(x, kernel_size=6).view(bsz, -1)
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires,
            bsz=bsz,
            device=x.device,
            record_op=True,
        )
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)

    def encode_to_quantum_vector(self, x: torch.Tensor) -> torch.Tensor:
        """Return the 16‑dimensional vector that will be fed to the encoder."""
        with torch.no_grad():
            pooled = F.avg_pool2d(x, kernel_size=6).view(x.shape[0], -1)
        return pooled


__all__ = ["QFCHybridModel"]
