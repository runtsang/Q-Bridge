"""Hybrid Nat Model: Quantum encoder with global and patch‑based quanvolution.

This module implements the quantum side of HybridNatModel.  It contains
two complementary quantum encoders:
    1. GlobalEncoder – a 4‑wire variational circuit that processes a
       global image embedding (average‑pooled features).
    2. PatchEncoder – a 4‑wire random kernel applied to every 2×2 image
       patch (the “quanvolution” idea).

The outputs of both encoders are averaged to produce a 4‑dimensional
feature vector that is batch‑normalised.  The design mirrors the
classical counterpart while adding a quantum kernel capable of
exploring high‑dimensional Hilbert spaces.

The architecture:
    Global path: encoder → random layer → measurement
    Patch path: per‑patch encoder → random layer → measurement
    Combine → BatchNorm1d
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class GlobalEncoder(tq.QuantumModule):
    """Variational circuit that consumes a 16‑dimensional vector."""

    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice, features: torch.Tensor) -> torch.Tensor:
        self.encoder(qdev, features)
        self.q_layer(qdev)
        return self.measure(qdev)


class PatchEncoder(tq.QuantumModule):
    """Random 4‑wire kernel applied to each 2×2 image patch."""

    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice, patch: torch.Tensor) -> torch.Tensor:
        self.encoder(qdev, patch)
        self.q_layer(qdev)
        return self.measure(qdev)


class HybridNatModel(tq.QuantumModule):
    """Quantum encoder that fuses a global variational circuit with a
    quanvolutional patch kernel.
    """

    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.global_encoder = GlobalEncoder()
        self.patch_encoder = PatchEncoder()
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device

        # ----- Global path -----
        qdev_global = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=device, record_op=True)
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        global_out = self.global_encoder(qdev_global, pooled)

        # ----- Patch path -----
        qdev_patch = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=device, record_op=True)
        patches = []
        img = x.view(bsz, 28, 28)
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = img[:, r:r + 2, c:c + 2].view(bsz, 4)
                meas = self.patch_encoder(qdev_patch, patch)
                patches.append(meas)
                qdev_patch.reset()  # clear state for next patch

        patch_out = torch.mean(torch.stack(patches, dim=0), dim=0)

        # Combine and normalise
        out = (global_out + patch_out) / 2
        return self.norm(out)


__all__ = ["HybridNatModel"]
