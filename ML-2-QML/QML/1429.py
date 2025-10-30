"""Hybrid quantum‑classical model with an extended variational circuit."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class QuantumNATExtended(tq.QuantumModule):
    """A quantum‑classical model that extends the original QFCModel by adding
    an entangling variational layer and a regularized measurement. The
    architecture still produces a 4‑dimensional output but now includes
    additional parameterized gates that enable richer quantum feature
    representations.

    The module can be used as a drop‑in replacement for the original QFCModel
    in quantum‑aware pipelines.
    """

    class QLayer(tq.QuantumModule):
        """Parameterised variational layer with entanglement and random gates."""

        def __init__(self) -> None:
            super().__init__()
            self.n_wires = 4
            self.random_layer = tq.RandomLayer(
                n_ops=50, wires=list(range(self.n_wires))
            )
            # Parameterised single‑qubit rotations
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)
            # Parameterised two‑qubit entangling gate
            self.crx = tq.CRX(has_params=True, trainable=True)
            # Additional entanglement pattern
            self.cnot = tq.CNOT()

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice) -> None:
            # Random initialisation
            self.random_layer(qdev)
            # Parameterised rotations on each qubit
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)
                self.rz(qdev, wires=wire)
            # Entanglement block
            self.crx(qdev, wires=[0, 1])
            self.crx(qdev, wires=[2, 3])
            self.cnot(qdev, wires=[1, 2])
            # Additional single‑qubit gates for diversity
            tqf.hadamard(qdev, wires=0, static=self.static_mode, parent_graph=self.graph)
            tqf.sx(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)

    def __init__(self) -> None:
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
        # Classical pooling to match the 4‑wire encoder
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)


__all__ = ["QuantumNATExtended"]
