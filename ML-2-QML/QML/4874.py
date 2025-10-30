"""Quantum‑aware fusion module mirroring the `QuanvolutionFusion` class."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
from torchquantum.functional import static_support
import networkx as nx

Tensor = torch.Tensor


class QuantumKernelQuantum(tq.QuantumModule):
    """Parameter‑free quantum kernel for a 2×2 patch."""
    def __init__(self, num_qubits: int = 4) -> None:
        super().__init__()
        self.n_wires = num_qubits
        self.qdevice = tq.QuantumDevice(self.n_wires)
        # fixed random circuit
        self.circuit = tq.RandomLayer(
            n_ops=8,
            wires=list(range(self.n_wires)),
        )

    @static_support
    def forward(self, qdev: tq.QuantumDevice, x: Tensor) -> Tensor:
        # x: (B, 2, 2)
        batch = x.shape[0]
        qdev.reset_states(batch)
        flat = x.view(batch, -1)  # (B,4)
        for i in range(self.n_wires):
            qdev.ry(flat[:, i], wires=i)
        self.circuit(qdev)
        amp = qdev.states[:, 0]
        return torch.abs(amp) ** 2  # (B,)


class StochasticQuanvolutionFilterQuantum(tq.QuantumModule):
    """Quantum counterpart of ``StochasticQuanvolutionFilter``."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.kernel = QuantumKernelQuantum()
        self.n_wires = 1
        self.summed_circuit = tq.RandomLayer(
            n_ops=4,
            wires=[0],
        )

    @static_support
    def forward(self, qdev: tq.QuantumDevice, x: Tensor) -> Tensor:
        # x: (B, 1, H, W)
        bsz, _, h, w = x.shape
        assert h % self.kernel_size == 0 and w % self.kernel_size == 0, (
            "Image dimensions must be divisible by kernel_size."
        )
        patches = x.unfold(2, self.kernel_size, self.kernel_size) \
                   .unfold(3, self.kernel_size, self.kernel_size)  # (B, 1, Nh, Nw, ks, ks)
        patches = patches.contiguous().view(bsz, -1, self.kernel_size, self.kernel_size)
        patch_outputs = self.kernel(qdev, patches.view(-1, self.kernel_size, self.kernel_size))  # (B*Np,)
        patch_outputs = patch_outputs.view(bsz, -1)  # (B, Np)
        qdev.reset_states(bsz)
        for i in range(patch_outputs.shape[1]):
            qdev.ry(patch_outputs[:, i], wires=0)
        self.summed_circuit(qdev)
        amp = qdev.states[:, 0]
        return torch.abs(amp) ** 2  # (B,)


class KernelMixingLayerQuantum(tq.QuantumModule):
    """Mixing layer that projects the quantum‑kernel Gram matrix to a lower space."""
    def __init__(self, in_features: int, out_features: int, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma
        self.out = nn.Linear(in_features, out_features)

    def forward(self, x: Tensor) -> Tensor:
        gram = torch.exp(-self.gamma * torch.cdist(x, x, p=2) ** 2)
        return self.out(gram @ x)


class GraphPrunerQuantum(tq.QuantumModule):
    """Same pruning logic as the classical version but implemented as a quantum module."""
    def __init__(self, weight: Tensor, threshold: float = 0.9) -> None:
        super().__init__()
        self.register_buffer("mask", self._build_mask(weight, threshold))

    def _build_mask(self, weight: Tensor, threshold: float) -> Tensor:
        return (weight.abs() >= threshold).float()

    def forward(self, x: Tensor) -> Tensor:
        return x * self.mask


class QuanvolutionFusionQuantum(tq.QuantumModule):
    """End‑to‑end quantum‑aware model that mirrors the classical `QuanvolutionFusion`."""
    def __init__(self, num_classes: int = 10, kernel_size: int = 2) -> None:
        super().__init__()
        self.qfilter = StochasticQuanvolutionFilterQuantum(kernel_size=kernel_size)
        self._flatten_dim = self._calc_flattened_dim(kernel_size)
        self.mix = KernelMixingLayerQuantum(self._flatten_dim, self._flatten_dim // 2)
        self.fc = nn.Linear(self._flatten_dim // 2, num_classes)
        self.pruner = GraphPrunerQuantum(self.fc.weight, threshold=0.95)

    def _calc_flattened_dim(self, kernel_size: int) -> int:
        dummy = torch.zeros(1, 1, 28, 28)
        out = self.qfilter(dummy)
        return out.shape[1]

    @static_support
    def forward(self, qdev: tq.QuantumDevice, x: Tensor) -> Tensor:
        x = self.qfilter(qdev, x)
        x = self.mix(x)
        x = self.pruner(x)
        logits = self.fc(x)
        return F.log_softmax(logits, dim=-1)


__all__ = [
    "QuantumKernelQuantum",
    "StochasticQuanvolutionFilterQuantum",
    "KernelMixingLayerQuantum",
    "GraphPrunerQuantum",
    "QuanvolutionFusionQuantum",
]
