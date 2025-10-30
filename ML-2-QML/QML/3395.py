"""QuantumHybridNet: Quantum module with parameterised variational circuit.

This module implements a hybrid network where the classical
convolutional backbone is replaced by a quantum expectation
layer.  The quantum part uses torchquantum to construct a
parameter‑efficient variational circuit with a tunable shift
rule.  The output is normalised via BatchNorm1d.

Author: gpt-oss-20b
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf

__all__ = ["QuantumHybridNet"]

class _QLayer(tq.QuantumModule):
    """
    Parameter‑efficient variational layer.
    Combines a random layer with trainable single‑qubit rotations
    and a controlled‑rotation gate.
    """
    def __init__(self, n_wires: int = 4, n_ops: int = 50):
        super().__init__()
        self.n_wires = n_wires
        self.random_layer = tq.RandomLayer(n_ops=n_ops, wires=list(range(self.n_wires)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)
        self.rz = tq.RZ(has_params=True, trainable=True)
        self.crx = tq.CRX(has_params=True, trainable=True)

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice) -> None:
        # Random layer to provide entanglement
        self.random_layer(qdev)
        # Trainable single‑qubit rotations
        self.rx(qdev, wires=0)
        self.ry(qdev, wires=1)
        self.rz(qdev, wires=2)
        # Controlled‑rotation entanglement
        self.crx(qdev, wires=[0, 3])
        # Additional static gates for expressivity
        tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
        tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
        tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)

class QuantumHybridNet(tq.QuantumModule):
    """
    Hybrid network that encodes classical features into a quantum state,
    applies a parameterised variational circuit, measures the
    expectation values, and normalises the result.
    """
    def __init__(self,
                 in_features: int,
                 n_wires: int = 4,
                 n_layers: int = 2,
                 num_classes: int = 2) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layers = nn.ModuleList([_QLayer(n_wires=self.n_wires) for _ in range(n_layers)])
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the quantum hybrid network.

        Parameters
        ----------
        x : torch.Tensor
            1‑D tensor of shape (batch, in_features) containing
            flattened classical features.

        Returns
        -------
        torch.Tensor
            Normalised logits of shape (batch, num_classes).
        """
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires,
                                bsz=bsz,
                                device=x.device,
                                record_op=True)
        # Encode the classical data
        self.encoder(qdev, x)
        # Apply variational layers
        for layer in self.q_layers:
            layer(qdev)
        # Measure expectation values
        out = self.measure(qdev)
        # Normalise
        return self.norm(out)
