"""Hybrid quanvolution filter using TorchQuantum.

This module mirrors the classical implementation but replaces the
convolutional filter with a quantum encoder that processes 2×2 image
patches.  The filter optionally uses a quantum kernel ansatz to
compute a similarity measure between patches and a learned prototype
vector.  The implementation is fully compatible with the original
QuanvolutionFilter/QuanvolutionClassifier API and can be swapped
into existing training pipelines.

The scaling paradigm is a *combination* of variational quantum
circuits and classical linear heads, enabling non‑linear feature
learning while keeping the final classification step classical.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
from torchquantum.functional import func_name_dict
from typing import Sequence


class KernalAnsatz(tq.QuantumModule):
    """
    Quantum kernel ansatz that encodes two input vectors into a
    shared circuit and measures overlap.

    Parameters
    ----------
    func_list : list[dict]
        List of gate specifications used for encoding.
    """

    def __init__(self, func_list):
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)


class Kernel(tq.QuantumModule):
    """
    Quantum kernel module that evaluates the overlap between two
    classical vectors via a fixed ansatz.
    """

    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = KernalAnsatz(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])


class QuanvolutionHybridFilter(tq.QuantumModule):
    """
    Quantum quanvolution filter that processes 2×2 patches of an image
    through a variational circuit and optionally a quantum kernel.

    Parameters
    ----------
    use_kernel : bool, default False
        If True, each patch is fed through the Kernel module to compute
        a similarity score against a learned prototype vector.
    """

    def __init__(self, use_kernel: bool = False):
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
        self.use_kernel = use_kernel
        if self.use_kernel:
            self.kernel = Kernel()
            # Prototype vector for kernel comparison
            self.register_buffer("prototype", torch.randn(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input image batch of shape (B, 1, 28, 28) or (B, 28, 28).

        Returns
        -------
        torch.Tensor
            Feature vector of shape (B, 4 * 14 * 14).  When use_kernel is
            True the 4‑dimensional patch representation is replaced by
            an overlap score against the prototype.
        """
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                data = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, data)
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                if self.use_kernel:
                    # Compute kernel against prototype for each batch element
                    kernel_vals = []
                    for i in range(bsz):
                        val = self.kernel(measurement[i].unsqueeze(0), self.prototype.unsqueeze(0))
                        kernel_vals.append(val)
                    kernel_vals = torch.stack(kernel_vals, dim=0)
                    patches.append(kernel_vals.view(bsz, 1))
                else:
                    patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)


class QuanvolutionHybridClassifier(nn.Module):
    """
    Hybrid classifier that uses the quantum quanvolution filter
    followed by a classical linear head.
    """

    def __init__(self, use_kernel: bool = False):
        super().__init__()
        self.qfilter = QuanvolutionHybridFilter(use_kernel)
        # Output dimension depends on whether kernel is used
        out_dim = 4 * 14 * 14 if not use_kernel else 1 * 14 * 14
        self.linear = nn.Linear(out_dim, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    """
    Compute the Gram matrix between two sequences of tensors using
    the quantum kernel defined above.

    Parameters
    ----------
    a, b : Sequence[torch.Tensor]
        Sequences of tensors to compare.  Each tensor is flattened
        before kernel evaluation.

    Returns
    -------
    np.ndarray
        Gram matrix of shape (len(a), len(b)).
    """
    kernel = Kernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])


__all__ = ["QuanvolutionHybridFilter", "QuanvolutionHybridClassifier", "kernel_matrix"]
