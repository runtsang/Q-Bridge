"""Quantum hybrid kernel that extends the classical RBF with a quantum kernel derived from a quantum fully‑connected feature extractor."""

from __future__ import annotations

import numpy as np
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
from torchquantum.functional import func_name_dict

class QFCModel(tq.QuantumModule):
    """Quantum fully‑connected model inspired by the Quantum‑NAT paper."""
    class QLayer(tq.QuantumModule):
        def __init__(self):
            super().__init__()
            self.n_wires = 4
            self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
            self.rx0 = tq.RX(has_params=True, trainable=True)
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.rz0 = tq.RZ(has_params=True, trainable=True)
            self.crx0 = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            self.rx0(qdev, wires=0)
            self.ry0(qdev, wires=1)
            self.rz0(qdev, wires=3)
            self.crx0(qdev, wires=[0, 2])
            func_name_dict["hadamard"](qdev, wires=3)
            func_name_dict["sx"](qdev, wires=2)
            func_name_dict["cnot"](qdev, wires=[3, 0])

    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)

class HybridQuantumKernel(tq.QuantumModule):
    """Combines a classical RBF kernel with a quantum kernel derived from QFCModel."""
    def __init__(self, alpha: float = 0.5, n_wires: int = 4):
        super().__init__()
        self.alpha = alpha
        self.qfc_model = QFCModel()
        self.n_wires = n_wires

    def _quantum_kernel_single(self, xi: torch.Tensor, yj: torch.Tensor) -> torch.Tensor:
        """Compute quantum kernel for a single pair of samples."""
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=1, device=xi.device)
        xi = xi[:self.n_wires]
        yj = yj[:self.n_wires]
        for i in range(self.n_wires):
            func_name_dict["ry"](qdev, wires=i, params=xi[i])
        for i in range(self.n_wires):
            func_name_dict["ry"](qdev, wires=i, params=-yj[i])
        return torch.abs(qdev.states.view(-1)[0])

    @tq.static_support
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute hybrid kernel between two batches of images."""
        x_flat = x.view(x.shape[0], -1)
        y_flat = y.view(y.shape[0], -1)
        diff = x_flat.unsqueeze(1) - y_flat.unsqueeze(0)
        rbf_val = torch.exp(-torch.sum(diff ** 2, dim=-1, keepdim=True))
        N, M = x_flat.shape[0], y_flat.shape[0]
        quantum_val = torch.zeros((N, M, 1), device=x.device, dtype=x.dtype)
        for i in range(N):
            for j in range(M):
                quantum_val[i, j, 0] = self._quantum_kernel_single(x_flat[i], y_flat[j])
        return self.alpha * rbf_val + (1.0 - self.alpha) * quantum_val

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], alpha: float = 0.5) -> np.ndarray:
    """Compute Gram matrix for two datasets of images using HybridQuantumKernel."""
    kernel = HybridQuantumKernel(alpha)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

__all__ = ["QFCModel", "HybridQuantumKernel", "kernel_matrix"]
