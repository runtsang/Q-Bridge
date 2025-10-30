"""Hybrid quantum kernel combining a variational RBF circuit with a quanvolution layer."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
from torchquantum.functional import op_name_dict


class QuantumRBFAnsatz(tq.QuantumModule):
    """Variational ansatz that encodes the difference between two data vectors."""

    def __init__(self, gamma: float = 1.0, n_wires: int = 4) -> None:
        super().__init__()
        self.gamma = gamma
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [i], "func": "ry", "wires": [i]}
                for i in range(n_wires)
            ]
        )
        self.var_layer = tq.RandomLayer(n_ops=8, wires=list(range(n_wires)))

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        # Encode x
        self.encoder(q_device, x)
        self.var_layer(q_device)
        # Encode -y
        neg_y = -y
        self.encoder(q_device, neg_y)
        self.var_layer(q_device)


class QuanvolutionFilter(tq.QuantumModule):
    """Quantum quanvolution layer that processes 2×2 patches of an image."""

    def __init__(self, n_wires: int = 4, n_ops: int = 8) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [i], "func": "ry", "wires": [i]}
                for i in range(n_wires)
            ]
        )
        self.layer = tq.RandomLayer(n_ops=n_ops, wires=list(range(n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, patch)
                self.layer(qdev)
                meas = self.measure(qdev)
                patches.append(meas.view(bsz, 4))
        return torch.cat(patches, dim=1)


class HybridKernel(tq.QuantumModule):
    """
    Hybrid quantum kernel that sums a variational RBF kernel with a quantum
    quanvolution feature kernel.  It can be used as a drop‑in replacement for
    classical kernels in quantum‑classical hybrid models.
    """

    def __init__(self, gamma: float = 1.0, n_wires: int = 4, n_ops: int = 8) -> None:
        super().__init__()
        self.rbf_ansatz = QuantumRBFAnsatz(gamma, n_wires)
        self.quanv = QuanvolutionFilter(n_wires, n_ops)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # RBF quantum kernel
        self.rbf_ansatz(self.q_device, x, y)
        rbf_val = torch.abs(self.q_device.states.view(-1)[0])

        # Quanvolution quantum kernel
        qx = self.quanv(x)
        qy = self.quanv(y)
        quanv_val = torch.sum(qx * qy, dim=-1, keepdim=True)

        return rbf_val + quanv_val

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """Compute Gram matrix for two collections of samples."""
        return np.array(
            [[self.forward(x, y).item() for y in b] for x in a]
        )


__all__ = ["QuantumRBFAnsatz", "QuanvolutionFilter", "HybridKernel"]
