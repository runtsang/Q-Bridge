"""Quantum estimator that mirrors the classical EstimatorQNN architecture.

The circuit encodes a 28×28 image into a 4‑qubit device using a
general 4‑qubit RyZXY encoder (from Quantum‑NAT), applies a
parameter‑ised QLayer composed of random rotations, single‑qubit gates
and a controlled‑RX, then measures all qubits in the Pauli‑Z basis.
The expectation values are batch‑normalised and can be used as a
regression target.  All parameters are trainable via the
tq.QuantumModule API.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class EstimatorQNN(tq.QuantumModule):
    """Quantum regression estimator inspired by EstimatorQNN and Quantum‑NAT."""

    class QLayer(tq.QuantumModule):
        """Core variational block with random layer and trainable gates."""

        def __init__(self) -> None:
            super().__init__()
            self.n_wires = 4
            # Random layer that produces a diverse initial circuit
            self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
            # Trainable single‑qubit rotations
            self.rx0 = tq.RX(has_params=True, trainable=True)
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.rz0 = tq.RZ(has_params=True, trainable=True)
            # Trainable controlled‑RX
            self.crx0 = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random_layer(qdev)
            self.rx0(qdev, wires=0)
            self.ry0(qdev, wires=1)
            self.rz0(qdev, wires=3)
            self.crx0(qdev, wires=[0, 2])
            # Additional deterministic gates
            tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
            tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
            tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)

    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        # Encoder that maps classical pixels to qubit rotations
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for a batch of images.

        Parameters
        ----------
        x:
            Tensor of shape (batch, 1, H, W) with pixel values in [0, 1].

        Returns
        -------
        torch.Tensor
            Normalised expectation values for each qubit.
        """
        bsz = x.shape[0]
        # Reduce to a 4×4 feature map (16 values) for the 4‑qubit encoder
        pooled = F.avg_pool2d(x, kernel_size=6).view(bsz, -1)
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True
        )
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)

__all__ = ["EstimatorQNN"]
