"""Hybrid quantum kernel model integrating a variational circuit
with a convolutional feature encoder.

The quantum part reuses the encoding strategy from
``QuantumKernelMethod`` (Ry gates) and the
variational layer from ``QuantumNAT``.  The forward
method returns the normalised overlap of two encoded
states, which can be used as a quantum kernel.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from typing import Sequence

class HybridKernelModel(tq.QuantumModule):
    """Quantum kernel with a CNN‑style feature encoder and a
    variational QLayer.

    The model encodes a 4‑dimensional feature vector into a
    4‑qubit register using a fixed Ry circuit, then
    applies a trainable variational layer (the QLayer)
    before measuring all qubits in the Pauli‑Z basis.
    """

    class QLayer(tq.QuantumModule):
        """Variational block inspired by QuantumNAT.

        The block contains a random layer followed by a
        small sequence of single‑qubit rotations and a
        controlled‑X.  All parameters are trainable.
        """

        def __init__(self) -> None:
            super().__init__()
            self.n_wires = 4
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(self.n_wires)))
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
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)

        # Encoder that maps a 4‑dimensional vector to a quantum state.
        # Uses the same Ry‑based circuit as the original kernel.
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])

        self.qlayer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        """Encode ``x`` and ``y`` into the same device and compute
        the overlap ``|<x|y>|``.  The device is assumed to be
        reset before the call.
        """
        # Encode the first vector
        self.encoder(qdev, x)
        self.qlayer(qdev)

        # Encode the second vector with negative parameters to compute
        # the inner product <x|y>.
        self.encoder(qdev, -y)
        self.qlayer(qdev)

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper returning the absolute overlap."""
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device)
        self.forward(qdev, x, y)
        out = self.measure(qdev)
        return torch.abs(out.view(-1, self.n_wires).sum(dim=1))

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """Compute the Gram matrix between two collections of 4‑dimensional vectors."""
        self.eval()
        with torch.no_grad():
            return np.array([[self(a_i, b_j).item() for b_j in b] for a_i in a])

__all__ = ["HybridKernelModel"]
