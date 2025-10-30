"""Quantum‑kernel variant of EstimatorQNN.

Uses TorchQuantum to encode data into a parameterised ansatz and evaluates the
overlap between two encoded states.  The kernel value is passed to a linear
read‑out layer, mirroring the classical kernel branch.
"""

from __future__ import annotations

import torch
from torch import nn
import torchquantum as tq
from torchquantum.functional import func_name_dict
from typing import Sequence


class QuantumKernel(tq.QuantumModule):
    """Fixed ansatz that maps classical data into a 4‑qubit state."""

    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Encode `x` then `y` (with opposite signs) and return the absolute
        overlap of the resulting two states.

        Parameters
        ----------
        x : torch.Tensor
            Shape (B, D)
        y : torch.Tensor
            Shape (1, D) – a single reference vector

        Returns
        -------
        torch.Tensor
            Kernel values of shape (B,)
        """
        q_device.reset_states(x.shape[0])
        # Encode `x`
        for i in range(self.n_wires):
            params = x[:, i] if tq.op_name_dict["ry"].num_params else None
            func_name_dict["ry"](q_device, wires=[i], params=params)
        # Encode `-y`
        for i in range(self.n_wires):
            params = -y[:, i] if tq.op_name_dict["ry"].num_params else None
            func_name_dict["ry"](q_device, wires=[i], params=params)
        # Return absolute overlap of the first state with the second
        return torch.abs(q_device.states.view(-1)[0])


class EstimatorQNN(nn.Module):
    """Hybrid estimator that uses a quantum kernel for feature mapping."""

    def __init__(
        self,
        input_dim: int = 2,
        n_wires: int = 4,
        output_dim: int = 1,
    ):
        super().__init__()
        self.kernel = QuantumKernel(n_wires)
        # Simple linear read‑out after kernel
        self.readout = nn.Linear(1, output_dim)

    def forward(self, x: torch.Tensor, x_ref: torch.Tensor | None = None) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input batch of shape (B, D).
        x_ref : torch.Tensor, optional
            Reference batch used for kernel evaluation.
            If omitted, a self‑kernel is computed.

        Returns
        -------
        torch.Tensor
            Predicted scalar per sample.
        """
        if x_ref is None:
            x_ref = x
        # Compute kernel matrix by looping over references
        batch_size = x.shape[0]
        ref_size = x_ref.shape[0]
        k = torch.empty(batch_size, ref_size, device=x.device)
        for i in range(ref_size):
            k[:, i] = self.kernel(x, x_ref[i : i + 1])
        # Mean similarity to all references
        feats = k.mean(dim=-1, keepdim=True)  # (B, 1)
        return self.readout(feats)

__all__ = ["EstimatorQNN"]
