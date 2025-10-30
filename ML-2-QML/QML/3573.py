"""Hybrid quantum model that fuses a classical encoder with a variational
self‑attention‑style circuit.  The circuit interleaves random operations,
parameterised RX/RY/RZ rotations, and CRX entangling gates, mimicking
the attention pattern of the classical counterpart.

The model:
1. Encodes a 16‑dim classical vector into a 4‑qubit quantum state.
2. Applies a variational layer that mixes random gates with
   self‑attention‑style rotations and entanglements.
3. Measures all qubits in the Z‑basis and normalises the result.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class QuantumNATHybrid(tq.QuantumModule):
    """
    Quantum implementation of the hybrid Quantum‑NAT model.
    The forward method takes a batch of images, pools them, encodes the
    pooled vector into a 4‑qubit state, runs a variational circuit that
    incorporates self‑attention‑style gates, and returns a normalised
    measurement vector.
    """

    class QLayer(tq.QuantumModule):
        """
        Variational layer that combines random operations with
        self‑attention‑style rotations and entangling CRX gates.
        """

        def __init__(self):
            super().__init__()
            self.n_wires = 4
            # Random layer for expressive entanglement
            self.random_layer = tq.RandomLayer(
                n_ops=50, wires=list(range(self.n_wires))
            )
            # Self‑attention‑style parameterised gates
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)
            self.crx = tq.CRX(has_params=True, trainable=True)
            # Additional random gates from the original QFCModel
            self.rx0 = tq.RX(has_params=True, trainable=True)
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.rz0 = tq.RZ(has_params=True, trainable=True)
            self.crx0 = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice) -> None:
            # Random entanglement
            self.random_layer(qdev)
            # Self‑attention‑style rotations and entanglement
            for i in range(self.n_wires):
                self.rx(qdev, wires=i)
                self.ry(qdev, wires=i)
                self.rz(qdev, wires=i)
            for i in range(self.n_wires - 1):
                self.crx(qdev, wires=[i, i + 1])
            # Additional gates from the original design
            self.rx0(qdev, wires=0)
            self.ry0(qdev, wires=1)
            self.rz0(qdev, wires=3)
            self.crx0(qdev, wires=[0, 2])
            tqf.hadamard(
                qdev,
                wires=3,
                static=self.static_mode,
                parent_graph=self.graph,
            )
            tqf.sx(
                qdev,
                wires=2,
                static=self.static_mode,
                parent_graph=self.graph,
            )
            tqf.cnot(
                qdev,
                wires=[3, 0],
                static=self.static_mode,
                parent_graph=self.graph,
            )

    def __init__(self):
        super().__init__()
        self.n_wires = 4
        # Encoder maps the 16‑dim pooled vector into a 4‑qubit state
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict["4x4_ryzxy"]
        )
        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, 1, H, W]  (e.g., 28×28 MNIST)
        Returns:
            [batch, 4]  normalised measurement vector
        """
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True
        )
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)


__all__ = ["QuantumNATHybrid"]
