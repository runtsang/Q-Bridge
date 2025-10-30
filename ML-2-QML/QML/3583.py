"""Hybrid quantum‑classical model based on the Quantum‑NAT architecture.

The quantum sub‑module is a variational circuit that
combines a random layer, a parameterised rotation block
and the original gate sequence from the reference.
The classical encoder is reused from the original
implementation.  The output is normalised with a
BatchNorm1d layer to match the output dimension.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class HybridQFCModel(tq.QuantumModule):
    """Quantum‑classical hybrid model that mirrors the classical
    HybridQFCModel but replaces the FC block with a variational
    quantum circuit.

    The circuit comprises:
        * a GeneralEncoder that maps the pooled image features into
          a 4‑qubit state,
        * a random layer for expressivity,
        * a parameterised rotation block (one Ry per qubit) that
          emulates the FCL quantum circuit,
        * the gate sequence from the original Quantum‑NAT paper,
        * a measurement of all qubits in the Pauli‑Z basis.
    """

    class QLayer(tq.QuantumModule):
        """Variational quantum layer used in HybridQFCModel."""

        def __init__(self) -> None:
            super().__init__()
            self.n_wires = 4
            self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
            # Parameterised rotation block: one Ry per qubit
            self.param_rot = self.ParametricRotations(self.n_wires)
            # Additional fixed gates from the original paper
            self.hadamard = tqf.hadamard
            self.sx = tqf.sx
            self.cnot = tqf.cnot

        class ParametricRotations(tq.QuantumModule):
            """Applies a trainable Ry gate on each qubit."""

            def __init__(self, n_wires: int) -> None:
                super().__init__()
                self.n_wires = n_wires
                self.rys = nn.ModuleList(
                    [tq.RY(has_params=True, trainable=True) for _ in range(n_wires)]
                )

            @tq.static_support
            def forward(self, qdev: tq.QuantumDevice) -> None:  # pragma: no cover
                for idx, ry in enumerate(self.rys):
                    ry(qdev, wires=idx)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice) -> None:  # pragma: no cover
            self.random_layer(qdev)
            self.param_rot(qdev)
            self.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
            self.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
            self.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)

    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        # 4×4 encoder expects input shape (bsz, 16)
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)

__all__ = ["HybridQFCModel"]
