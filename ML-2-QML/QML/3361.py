"""Quantum‑enhanced convolutional network combining a quantum convolutional layer
with a fully‑connected quantum block inspired by the Quantum‑NAT architecture."""

from __future__ import annotations

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf


class QConvLayer(tq.QuantumModule):
    """Quantum convolutional block that processes each 2×2 patch."""
    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.random = tq.RandomLayer(n_ops=20, wires=list(range(n_wires)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)
        self.rz = tq.RZ(has_params=True, trainable=True)
        self.crx = tq.CRX(has_params=True, trainable=True)

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice) -> None:
        self.random(qdev)
        self.rx(qdev, wires=0)
        self.ry(qdev, wires=1)
        self.rz(qdev, wires=3)
        self.crx(qdev, wires=[0, 2])
        tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
        tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
        tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)


class QConvGen304(tq.QuantumModule):
    """Full quantum model that first applies a quantum convolutional layer
    to 2×2 patches and then passes the resulting feature map through a
    quantum fully‑connected block, mirroring the classical ConvGen304 architecture."""
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.conv_layer = QConvLayer(n_wires=self.n_wires)
        self.q_fc = QConvLayer(n_wires=self.n_wires)  # acts as the fully connected block
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, _, h, w = x.shape
        # Number of 2×2 patches
        n_patches_h = h // 2
        n_patches_w = w // 2
        n_patches = n_patches_h * n_patches_w

        # Create quantum device for all patches
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz * n_patches,
                                device=x.device, record_op=True)

        # Extract 2×2 patches and flatten
        patches = x.unfold(2, 2, 2).unfold(3, 2, 2)  # shape: bsz,1,Nh,Nw,2,2
        patches = patches.permute(0, 1, 2, 3, 4, 5).reshape(-1, 4)

        # Encode each patch into the quantum device
        self.conv_layer.encoder(qdev, patches)
        self.conv_layer(qdev)
        conv_out = self.measure(qdev)  # shape: (bsz*n_patches, 4)

        # Reshape back to spatial feature map
        conv_out = conv_out.reshape(bsz, n_patches_h, n_patches_w, self.n_wires)
        conv_out = conv_out.mean(dim=[2, 3])  # aggregate over spatial positions -> (bsz, 4)

        # Fully connected quantum block
        qdev_fc = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz,
                                   device=x.device, record_op=True)
        self.q_fc.encoder(qdev_fc, conv_out)
        self.q_fc(qdev_fc)
        fc_out = self.measure(qdev_fc)

        return self.norm(fc_out)


__all__ = ["QConvGen304"]
