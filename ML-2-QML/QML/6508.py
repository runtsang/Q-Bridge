"""Hybrid fully‑connected layer with a quantum circuit.

This module implements the same high‑level architecture as the classical
counterpart but replaces the final linear block with a variational
quantum circuit.  The quantum circuit is built with TorchQuantum and
inherits all differentiability properties, enabling end‑to‑end
training with back‑propagation through the quantum device.

Key design choices
------------------
* A 4‑wire quantum device, matching the number of output features in
  the classical model.
* A RandomLayer for entanglement, followed by trainable RX/RZ/CRX gates.
* A GeneralEncoder that maps the pooled classical features into the
  quantum state via a 4×4 RyZXY pattern.
* Measurement of all wires in the Pauli‑Z basis, producing a real‑valued
  vector that is batch‑normalised.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from typing import List


class HybridFCL(tq.QuantumModule):
    """
    Quantum counterpart of the hybrid fully‑connected layer.

    Architecture
    ------------
    * Classical CNN encoder (identical to HybridFCL in the ML module).
    * Average pooling to reduce spatial dimension.
    * GeneralEncoder to map the pooled features into a 4‑wire quantum
      state.
    * QuantumLayer performing a RandomLayer + RX/RZ/CRX gates.
    * Measurement of all wires (Pauli‑Z) followed by batch normalisation.
    """

    class QuantumLayer(tq.QuantumModule):
        def __init__(self) -> None:
            super().__init__()
            self.n_wires = 4
            self.random_layer = tq.RandomLayer(
                n_ops=50, wires=list(range(self.n_wires))
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
            # Optional static gates to enrich expressibility
            tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
            tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
            tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)

    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict["4x4_ryzxy"]
        )
        self.q_layer = self.QuantumLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the hybrid quantum layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, n_wires).
        """
        bsz = x.shape[0]
        # Classical encoder identical to the ML version
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True
        )
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)


__all__ = ["HybridFCL"]
