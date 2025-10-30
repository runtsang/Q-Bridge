"""Quantum‑NAT inspired variational model with a flexible encoder.

The quantum model mirrors the classical interface but replaces the
convolutional feature extractor with a trainable variational circuit.
A selectable encoder (default 4×4‑RYZXY) feeds the pre‑processed
features into a deep parameterised block consisting of random
layer, RZ/RX rotations, and entangling CRX gates.  The measurement
operator can be swapped for any Pauli product, and the output is
normalised with a BatchNorm1d layer.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class QuantumNATModel(tq.QuantumModule):
    """Variational Quantum Circuit for the Quantum‑NAT task.

    Parameters
    ----------
    n_wires : int, optional
        Number of qubit wires. Defaults to 4.
    encoder_name : str, optional
        Name of the encoder from `tq.encoder_op_list_name_dict`. Defaults
        to ``"4x4_ryzxy"``.
    measure_operator : str, optional
        Pauli operator for measurement. One of ``"PauliZ"``, ``"PauliX"``,
        ``"PauliY"``. Defaults to ``"PauliZ"``.
    """

    class VariationalBlock(tq.QuantumModule):
        """Deep trainable block with random layer and entangling gates."""

        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires
            # 50 random single‑qubit ops per layer
            self.random_layer = tq.RandomLayer(
                n_ops=50, wires=list(range(self.n_wires))
            )
            # Trainable rotations
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)
            # Entangling gates
            self.crx = tq.CRX(has_params=True, trainable=True)
            self.cnot = tq.CNOT(has_params=True, trainable=False)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random_layer(qdev)
            self.rx(qdev, wires=0)
            self.ry(qdev, wires=1)
            self.rz(qdev, wires=2)
            self.crx(qdev, wires=[0, 3])
            self.cnot(qdev, wires=[2, 1])
            tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
            tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)

    def __init__(
        self,
        n_wires: int = 4,
        encoder_name: str = "4x4_ryzxy",
        measure_operator: str = "PauliZ",
    ) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict[encoder_name]
        )
        self.var_block = self.VariationalBlock(n_wires)
        # Measurement
        if measure_operator == "PauliZ":
            self.measure = tq.MeasureAll(tq.PauliZ)
        elif measure_operator == "PauliX":
            self.measure = tq.MeasureAll(tq.PauliX)
        elif measure_operator == "PauliY":
            self.measure = tq.MeasureAll(tq.PauliY)
        else:
            raise ValueError(f"Unsupported operator: {measure_operator}")
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True
        )
        # Reduce the image to a 16‑dim vector (4×4)
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.var_block(qdev)
        out = self.measure(qdev)
        return self.norm(out)


__all__ = ["QuantumNATModel"]
