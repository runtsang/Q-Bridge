"""Quantum variant of QFCModel with interchangeable variational ansatz and dual measurement heads."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class QFCModel(tq.QuantumModule):
    """Quantum model inspired by Quantum‑NAT with two measurement heads.

    Parameters
    ----------
    n_wires : int
        Number of qubits in the device.
    ansatz_type : str
        Either ``"fixed"`` or ``"trainable"``.  ``"fixed"`` uses a
        pre‑defined random layer; ``"trainable"`` uses a variational
        ansatz with trainable rotation angles.
    """

    class VariationalLayer(tq.QuantumModule):
        """Variational ansatz that can be fixed or trainable."""

        def __init__(self, n_wires: int, trainable: bool):
            super().__init__()
            self.n_wires = n_wires
            self.trainable = trainable
            self.random = tq.RandomLayer(
                n_ops=40, wires=list(range(n_wires)), has_params=False
            )
            self.rx = tq.RX(has_params=True, trainable=trainable)
            self.ry = tq.RY(has_params=True, trainable=trainable)
            self.rz = tq.RZ(has_params=True, trainable=trainable)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            if self.trainable:
                for w in range(self.n_wires):
                    self.rx(qdev, wires=w)
                    self.ry(qdev, wires=w)
                    self.rz(qdev, wires=w)
            else:
                self.random(qdev)

    def __init__(self, n_wires: int = 4, ansatz_type: str = "fixed") -> None:
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.variational = self.VariationalLayer(
            n_wires, trainable=(ansatz_type == "trainable")
        )
        self.measure1 = tq.MeasureAll(tq.PauliZ)
        self.measure2 = tq.MeasureAll(tq.PauliX)
        self.norm = nn.BatchNorm1d(n_wires)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return two measurement results (Z and X bases)."""
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True
        )
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.variational(qdev)

        out1 = self.measure1(qdev)
        out2 = self.measure2(qdev)
        return self.norm(out1), self.norm(out2)

__all__ = ["QFCModel"]
