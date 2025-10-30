"""Hybrid quanvolution model using TorchQuantum kernels and a quantum self‑attention
block.  The filter encodes 2×2 image patches into a 4‑qubit state, evaluates a
fixed quantum kernel against a learnable template, and then applies a
quantum‑derived attention mechanism before a classical linear classifier.

This preserves the overall dimensionality of the classical version while
providing a fully quantum feature extraction pipeline that can be executed
on a simulator or real device via TorchQuantum.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
from torchquantum.functional import (
    func_name_dict,
    ry,
    rz,
    measure_all,
    PauliZ,
)


class KernalAnsatz(tq.QuantumModule):
    """Encodes classical data through a programmable list of quantum gates."""

    def __init__(self, func_list):
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = (
                x[:, info["input_idx"]]
                if tq.op_name_dict[info["func"]].num_params
                else None
            )
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = (
                -y[:, info["input_idx"]]
                if tq.op_name_dict[info["func"]].num_params
                else None
            )
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)


class Kernel(tq.QuantumModule):
    """Quantum kernel evaluated via a fixed TorchQuantum ansatz."""

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


class QuantumSelfAttention(tq.QuantumModule):
    """Basic quantum circuit representing a self‑attention style block."""

    def __init__(self, n_qubits: int = 4):
        super().__init__()
        self.n_qubits = n_qubits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape (B, seq_len, 4)
        batch, seq_len, _ = x.shape
        attentions = []
        for i in range(batch):
            qdev = tq.QuantumDevice(self.n_qubits)
            # encode each element of the sequence into separate qubits
            for j in range(seq_len):
                # toy encoding: use first two components as rotation angles
                ry(qdev, wires=j, params=x[i, j, 0])
                rz(qdev, wires=j, params=x[i, j, 1])
            meas = measure_all(qdev, PauliZ)
            attentions.append(meas.unsqueeze(0))
        return torch.cat(attentions, dim=0)  # (B, seq_len)


class HybridQuanvolution(tq.QuantumModule):
    """Quantum hybrid quanvolution with kernel‑based patch encoding and quantum attention."""

    def __init__(self, embed_dim: int = 4, num_heads: int = 2) -> None:
        super().__init__()
        self.kernel = Kernel()
        self.attention = QuantumSelfAttention(n_qubits=embed_dim)
        self.linear = nn.Linear(embed_dim * 14 * 14, 10)

    def _extract_patches(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 28, 28)
        patches = x.unfold(2, 2, 2).unfold(3, 2, 2)  # (B, 1, 14, 14, 2, 2)
        patches = patches.contiguous().view(x.shape[0], -1, 4)  # (B, 196, 4)
        return patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = self._extract_patches(x)  # (B, 196, 4)
        batch, seq_len, feat_dim = patches.shape
        # compute kernel values between each patch and a learnable template
        template = torch.zeros(feat_dim, device=x.device)
        kernel_values = []
        for i in range(seq_len):
            patch_vec = patches[:, i, :].view(batch, -1)
            k = self.kernel(patch_vec, template.expand(batch, -1))
            kernel_values.append(k)
        kernel_values = torch.stack(kernel_values, dim=1)  # (B, 196, 1)
        # apply quantum attention to the kernel values
        attn_output = self.attention(kernel_values)  # (B, 196)
        # flatten
        flat = attn_output.view(x.shape[0], -1)
        logits = self.linear(flat)
        return F.log_softmax(logits, dim=-1)


__all__ = ["HybridQuanvolution"]
