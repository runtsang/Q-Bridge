"""Hybrid quantum‑classical model with a variational circuit that incorporates classical feedback."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class QFCModelEnhanced(tq.QuantumModule):
    """Quantum‑enhanced version of QFCModelEnhanced.

    The model encodes the pooled classical features into rotation angles
    of a 4‑qubit variational circuit. After a random layer and a
    parameterised rotation block, the circuit is measured. The measurement
    results are fed back as additional parameters for a second rotation
    block, creating a controlled quantum‑classical loop. The final
    measurement vector is normalised and passed through a linear head
    to produce the four output logits.
    """

    class QLayer(tq.QuantumModule):
        def __init__(self):
            super().__init__()
            self.n_wires = 4
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(self.n_wires)))
            # First rotation block (parameterised by classical features)
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)
            # Second rotation block (feedback from measurement)
            self.rx_fb = tq.RX(has_params=True, trainable=True)
            self.ry_fb = tq.RY(has_params=True, trainable=True)
            self.rz_fb = tq.RZ(has_params=True, trainable=True)
            # Entanglement
            self.cnot = tq.CNOT

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice, params: torch.Tensor, fb_params: torch.Tensor):
            # Encode classical parameters
            self.random_layer(qdev)
            self.rx(qdev, wires=0, params=params[:, 0])
            self.ry(qdev, wires=1, params=params[:, 1])
            self.rz(qdev, wires=2, params=params[:, 2])
            self.rx_fb(qdev, wires=0, params=fb_params[:, 0])
            self.ry_fb(qdev, wires=1, params=fb_params[:, 1])
            self.rz_fb(qdev, wires=2, params=fb_params[:, 2])

            # Entangle
            self.cnot(qdev, wires=[0, 3])
            self.cnot(qdev, wires=[1, 2])

    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)
        # Classical head after measurement
        self.classical_head = nn.Sequential(
            nn.Linear(self.n_wires, 4)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)

        # Classical pooling to obtain 16‑dim feature vector
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)

        # Encode features into rotations
        self.encoder(qdev, pooled)

        # First rotation block uses first 3 features as angles
        params = pooled[:, :3]
        # Placeholder for feedback parameters (initially zeros)
        fb_params = torch.zeros_like(params)

        # Run variational circuit
        self.q_layer(qdev, params, fb_params)

        # Measurement
        out = self.measure(qdev)

        # Normalise measurement
        out = self.norm(out)

        # Classical head
        out = self.classical_head(out)
        return out


__all__ = ["QFCModelEnhanced"]
