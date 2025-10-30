"""Quantum module embedding a variational autoencoder style circuit and feature map."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler as Sampler

class QuantumNATHybrid(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self):
            super().__init__()
            self.n_latent = 3
            self.n_trash = 2
            self.n_aux = 1
            self.n_wires = self.n_latent + 2 * self.n_trash + self.n_aux
            self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)
            self.crx = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            # Variational gates on latent+trash qubits
            for w in range(self.n_latent + self.n_trash):
                self.rx(qdev, wires=w)
                self.ry(qdev, wires=w)
                self.rz(qdev, wires=w)
            # Swap‑test with auxiliary qubit
            aux_wire = self.n_latent + 2 * self.n_trash
            tqf.hadamard(qdev, wires=aux_wire)
            for i in range(self.n_trash):
                tqf.cnot(qdev, wires=[aux_wire, self.n_latent + i])
                tqf.cnot(qdev, wires=[aux_wire, self.n_latent + self.n_trash + i])
            tqf.hadamard(qdev, wires=aux_wire)

    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
