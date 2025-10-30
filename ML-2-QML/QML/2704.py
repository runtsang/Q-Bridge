"""Quantum kernel that encodes CNN features and evaluates overlap.

This implementation merges the quantum ansatz from QuantumKernelMethod
with the quantum feature map of QuantumNAT.  A classical CNN extracts
features, which are then encoded into a 4‑qubit device.  The quantum
circuit consists of a RandomLayer, trainable single‑qubit rotations
and a few two‑qubit gates.  The kernel value is the absolute
overlap of the states produced by encoding two images.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class Kernel(tq.QuantumModule):
    """Hybrid quantum kernel with classical CNN feature extractor."""

    def __init__(self, n_wires: int = 4, gamma: float = 1.0) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.gamma = gamma
        # Classical CNN part (same as ML version)
        self.features = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        # Quantum part
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)
        self.rz = tq.RZ(has_params=True, trainable=True)
        self.crx = tq.CRX(has_params=True, trainable=True)

    def _encode_features(self, qdev: tq.QuantumDevice, x: torch.Tensor, sign: int = 1) -> None:
        """Encode the pooled image features into the quantum device."""
        pooled = F.avg_pool2d(x, 6).view(x.shape[0], -1)
        # Apply sign to the feature vector
        self.features(qdev, sign * pooled)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return kernel value for two batches of images."""
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz,
                                device=x.device, record_op=True)
        # Encode first image features and quantum circuit
        self._encode_features(qdev, x, sign=1)
        self.random_layer(qdev)
        self.rx(qdev, wires=0)
        self.ry(qdev, wires=1)
        self.rz(qdev, wires=3)
        self.crx(qdev, wires=[0, 2])
        tqf.hadamard(qdev, wires=3)
        tqf.sx(qdev, wires=2)
        tqf.cnot(qdev, wires=[3, 0])

        # Encode second image features with negative sign and reversed circuit
        self._encode_features(qdev, y, sign=-1)
        self.crx(qdev, wires=[0, 2])
        tqf.sx(qdev, wires=2)
        tqf.hadamard(qdev, wires=3)
        self.rz(qdev, wires=3)
        self.ry(qdev, wires=1)
        self.rx(qdev, wires=0)
        self.random_layer(qdev)

        # Overlap measurement: absolute value of the first amplitude
        return torch.abs(qdev.states.view(-1)[0])

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """Compute Gram matrix using the quantum hybrid kernel."""
    kernel = Kernel(gamma=gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

__all__ = ["Kernel", "kernel_matrix"]
