"""Hybrid quantum‑classical model that uses a variational circuit and a classical sampler MLP."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class QFCModel(tq.QuantumModule):
    """Quantum feature extractor with a variational layer and a post‑processing sampler network."""

    class QLayer(tq.QuantumModule):
        """Variational sub‑module containing random gates and trainable rotations."""

        def __init__(self) -> None:
            super().__init__()
            self.n_wires = 4
            self.random_layer = tq.RandomLayer(
                n_ops=50,
                wires=list(range(self.n_wires)),
                seed=42,
            )
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
            tqf.hadamard(qdev, wires=3)
            tqf.sx(qdev, wires=2)
            tqf.cnot(qdev, wires=[3, 0])

    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        # Encoder that maps classical features into a quantum state
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer()
        # Measure all qubits in the Z basis
        self.measure = tq.MeasureAll(tq.PauliZ)
        # Classical sampler MLP that turns expectation values into a probability distribution
        self.sampler = nn.Sequential(
            nn.Linear(self.n_wires, 2),
            nn.Tanh(),
            nn.Linear(2, 4),
        )
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires,
            bsz=bsz,
            device=x.device,
            record_op=True,
        )
        # Reduce spatial resolution to match the encoder input size
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        # Encode classical features into the quantum device
        self.encoder(qdev, pooled)
        # Apply variational layer
        self.q_layer(qdev)
        # Obtain expectation values of Z operators
        z_exp = self.measure(qdev)
        # Convert expectation values to a probability distribution via the sampler MLP
        probs = F.softmax(self.sampler(z_exp), dim=-1)
        return self.norm(probs)


__all__ = ["QFCModel"]
