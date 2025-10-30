"""Hybrid quantum model using torchquantum that augments the classical CNN features
with a variational circuit. The architecture is a synthesis of the Quantum‑NAT
quantum module and the EstimatorQNN variational circuit design.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class HybridNATModel(tq.QuantumModule):
    """
    Quantum hybrid module.

    The module encodes a pooled image feature vector into a 4‑qubit
    variational circuit, measures Pauli‑Z, and applies a linear readout.
    The circuit design borrows from the QLayer of Quantum‑NAT and the
    single‑qubit EstimatorQNN circuit.
    """

    class QLayer(tq.QuantumModule):
        """Variational sub‑module."""

        def __init__(self) -> None:
            super().__init__()
            self.n_wires = 4
            self.random_layer = tq.RandomLayer(
                n_ops=50, wires=list(range(self.n_wires))
            )
            # Trainable single‑qubit rotations
            self.rx0 = tq.RX(has_params=True, trainable=True)
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.rz0 = tq.RZ(has_params=True, trainable=True)
            self.crx0 = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice) -> None:
            # Random mixing
            self.random_layer(qdev)
            # Parameterised rotations
            self.rx0(qdev, wires=0)
            self.ry0(qdev, wires=1)
            self.rz0(qdev, wires=3)
            self.crx0(qdev, wires=[0, 2])
            # Additional gates to enrich entanglement
            tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
            tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
            tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)

    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        # Encoder that maps 16‑dim input to rotations on 4 qubits
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict["4x4_ryzxy"]
        )
        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        # Linear readout to produce a scalar output
        self.readout = nn.Linear(self.n_wires, 1)
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input image batch of shape (B, C, H, W).

        Returns
        -------
        torch.Tensor
            Scalar prediction per sample.
        """
        bsz = x.shape[0]
        # Pool the image to a 16‑dim vector as in the original
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True
        )
        # Encode classical features into quantum parameters
        self.encoder(qdev, pooled)
        # Variational circuit
        self.q_layer(qdev)
        # Measure all qubits
        out = self.measure(qdev)
        # Linear readout
        out = self.readout(out)
        # Normalise
        return self.norm(out)


__all__ = ["HybridNATModel"]
