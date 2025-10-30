"""Quantum hybrid kernel‑attention model using TorchQuantum."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict


# --------------------------------------------------------------------------- #
# 1. Fraud‑detection inspired quantum layer
# --------------------------------------------------------------------------- #
@dataclass
class FraudLayerParameters:
    """Parameters for a photonic‑style fraud‑detection block."""
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]


class QuantumFraudLayer(tq.QuantumModule):
    """Quantum analogue of the classical fraud‑detection layer."""
    def __init__(self, params: FraudLayerParameters):
        super().__init__()
        self.params = params
        self.n_wires = 2
        self.qdev = tq.QuantumDevice(n_wires=self.n_wires)

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice) -> None:
        # Encode BS‑gate (beam splitter) style rotation
        func_name_dict["rx"](qdev, wires=[0], params=self.params.bs_theta)
        func_name_dict["ry"](qdev, wires=[1], params=self.params.bs_phi)
        # Apply phases
        for i, phase in enumerate(self.params.phases):
            func_name_dict["rz"](qdev, wires=[i], params=phase)
        # Squeezing (simulated with Rx for demo)
        for i, r in enumerate(self.params.squeeze_r):
            func_name_dict["rx"](qdev, wires=[i], params=r)
        # Displacement (simulated with Ry)
        for i, r in enumerate(self.params.displacement_r):
            func_name_dict["ry"](qdev, wires=[i], params=r)
        # Kerr (simulated with Rz)
        for i, k in enumerate(self.params.kerr):
            func_name_dict["rz"](qdev, wires=[i], params=k)


# --------------------------------------------------------------------------- #
# 2. Quantum self‑attention module
# --------------------------------------------------------------------------- #
class QuantumSelfAttention(tq.QuantumModule):
    """Variational self‑attention kernel implemented with controlled rotations."""
    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.random_layer = tq.RandomLayer(n_ops=20, wires=list(range(self.n_wires)))

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice) -> None:
        self.random_layer(qdev)
        for i in range(self.n_wires):
            func_name_dict["rx"](qdev, wires=[i], params=0.1 * i)
            func_name_dict["ry"](qdev, wires=[i], params=0.2 * i)
            func_name_dict["rz"](qdev, wires=[i], params=0.3 * i)
        # Entangling CRX style gates
        for i in range(self.n_wires - 1):
            func_name_dict["crx"](qdev, wires=[i, i + 1], params=0.4)


# --------------------------------------------------------------------------- #
# 3. CNN‑inspired quantum encoder (Quantum‑NAT style)
# --------------------------------------------------------------------------- #
class QFCQuantumModel(tq.QuantumModule):
    """Quantum CNN + FC projection using a variational encoder."""
    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = QuantumSelfAttention(n_wires=self.n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = torch.nn.BatchNorm1d(self.n_wires)

    @tq.static_support
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device)
        # Flatten image to 16‑D vector (4x4)
        pooled = torch.nn.functional.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)


# --------------------------------------------------------------------------- #
# 4. Hybrid quantum kernel‑attention model
# --------------------------------------------------------------------------- #
class HybridQuantumKernel(tq.QuantumModule):
    """
    Quantum analogue of HybridKernelAttentionModel:
        1. Encodes raw images with a variational CNN‑style encoder.
        2. Applies a fraud‑detection style layer.
        3. Runs a quantum self‑attention kernel.
        4. Returns an inner‑product similarity matrix.
    """
    def __init__(self, n_wires: int = 4, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma
        self.cnn_q = QFCQuantumModel(n_wires=n_wires)
        # Example fraud params (identity‑like)
        example_params = FraudLayerParameters(
            bs_theta=0.0, bs_phi=0.0,
            phases=(0.0, 0.0),
            squeeze_r=(1.0, 1.0),
            squeeze_phi=(0.0, 0.0),
            displacement_r=(0.0, 0.0),
            displacement_phi=(0.0, 0.0),
            kerr=(0.0, 0.0),
        )
        self.fraud_layer = QuantumFraudLayer(example_params)
        self.self_attn = QuantumSelfAttention(n_wires=n_wires)

    @tq.static_support
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the quantum kernel between x and y.

        Args:
            x: Tensor of shape (N, C, H, W)
            y: Tensor of shape (M, C, H, W)

        Returns:
            Kernel matrix of shape (N, M) on the current device.
        """
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.cnn_q.n_wires,
                                bsz=bsz,
                                device=x.device)
        # Encode x
        self.cnn_q.encoder(qdev, torch.nn.functional.avg_pool2d(x, 6).view(bsz, 16))
        self.fraud_layer(qdev)
        self.self_attn(qdev)
        out_x = self.cnn_q.measure(qdev)

        # Encode y
        qdev_y = tq.QuantumDevice(n_wires=self.cnn_q.n_wires,
                                  bsz=y.shape[0],
                                  device=y.device)
        self.cnn_q.encoder(qdev_y, torch.nn.functional.avg_pool2d(y, 6).view(y.shape[0], 16))
        self.fraud_layer(qdev_y)
        self.self_attn(qdev_y)
        out_y = self.cnn_q.measure(qdev_y)

        # Compute cosine‑like similarity (inner product)
        kernel = torch.abs(torch.einsum("bi,bj->ij", out_x, out_y))
        return kernel

# --------------------------------------------------------------------------- #
# 5. Gram matrix helper
# --------------------------------------------------------------------------- #
def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """
    Compute the quantum kernel Gram matrix between two datasets.

    Args:
        a: Iterable of tensors (N, C, H, W)
        b: Iterable of tensors (M, C, H, W)
        gamma: Scaling factor for the RBF‑style exponent.

    Returns:
        NumPy array of shape (N, M)
    """
    kernel = HybridQuantumKernel(gamma=gamma)
    return np.array([kernel(a[i], b[j]).cpu().numpy() for i in range(len(a)) for j in range(len(b))])


__all__ = [
    "FraudLayerParameters",
    "QuantumFraudLayer",
    "QuantumSelfAttention",
    "QFCQuantumModel",
    "HybridQuantumKernel",
    "kernel_matrix",
]
