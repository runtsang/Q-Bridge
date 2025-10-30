"""Hybrid quantum kernel combining amplitude‑overlap and sampler kernels.

The implementation mirrors the classical counterpart: a weighted sum
(`alpha`) of an analytical amplitude‑overlap kernel and a probability
dot‑product kernel derived from a parameterised quantum sampler.  The
sampler is a small 2‑qubit variational circuit with separate input and
weight parameters, enabling richer feature maps than the baseline
ansatz.  This dual‑kernel design allows systematic exploration of
quantum scaling versus classical scaling in downstream models.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict

__all__ = ["SamplerQNNQuantum", "HybridKernel"]


class SamplerQNNQuantum(tq.QuantumModule):
    """2‑qubit variational sampler used to produce a probability vector.

    The circuit consists of two layers of Ry gates driven by input
    parameters, a CX entangling gate, followed by a second layer driven
    by weight parameters and another CX.  Output probabilities are
    returned via the device state vector.
    """

    def __init__(self, weight_dim: int = 4) -> None:
        super().__init__()
        self.n_wires = 2
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.weight_dim = weight_dim

    @tq.static_support
    def forward(
        self,
        q_device: tq.QuantumDevice,
        x: torch.Tensor,
        w: torch.Tensor,
    ) -> None:
        """
        Execute the sampler circuit.

        Parameters
        ----------
        q_device : QuantumDevice
            Target device.
        x : Tensor
            Input parameters of shape (batch, 2).
        w : Tensor
            Weight parameters of shape (batch, weight_dim).
        """
        q_device.reset_states(x.shape[0])
        # first layer – input Ry gates
        for idx in range(2):
            tq.ry(q_device, wires=[idx], params=x[:, idx])
        # entangle
        tq.cx(q_device, wires=[0, 1])
        # second layer – weight Ry gates
        for idx in range(self.weight_dim):
            wire = idx % self.n_wires
            tq.ry(q_device, wires=[wire], params=w[:, idx])
        # final entanglement
        tq.cx(q_device, wires=[0, 1])


class HybridKernel(tq.QuantumModule):
    """Weighted combination of an amplitude‑overlap kernel and a sampler
    probability kernel.  The parameter ``alpha`` controls the mix.
    """

    def __init__(self, n_wires: int = 4, alpha: float = 0.5) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        # amplitude‑overlap ansatz
        self.ansatz = [
            {"input_idx": [0], "func": "ry", "wires": [0]},
            {"input_idx": [1], "func": "ry", "wires": [1]},
            {"input_idx": [2], "func": "ry", "wires": [2]},
            {"input_idx": [3], "func": "ry", "wires": [3]},
        ]
        self.sampler = SamplerQNNQuantum()
        self.alpha = alpha

    @tq.static_support
    def forward(
        self,
        q_device: tq.QuantumDevice,
        x: torch.Tensor,
        y: torch.Tensor,
        w_x: torch.Tensor,
        w_y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute a combined kernel value for two samples.

        Parameters
        ----------
        x, y : Tensor
            Input parameter vectors of shape (batch, 4).
        w_x, w_y : Tensor
            Weight vectors for the sampler of shape (batch, 4).
        """
        # amplitude‑overlap part
        q_device.reset_states(x.shape[0])
        for info in self.ansatz:
            params = (
                x[:, info["input_idx"][0]]
                if tq.op_name_dict[info["func"]].num_params
                else None
            )
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.ansatz):
            params = (
                -y[:, info["input_idx"][0]]
                if tq.op_name_dict[info["func"]].num_params
                else None
            )
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        amp_kernel = torch.abs(q_device.states.view(-1)[0])

        # sampler part
        sampler_device = tq.QuantumDevice(n_wires=2)
        self.sampler.forward(sampler_device, x, w_x)
        probs_x = sampler_device.states.view(-1)[0].abs()
        self.sampler.forward(sampler_device, y, w_y)
        probs_y = sampler_device.states.view(-1)[0].abs()
        samp_kernel = torch.sum(probs_x * probs_y, dim=-1, keepdim=True)

        # weighted combination
        return self.alpha * amp_kernel + (1.0 - self.alpha) * samp_kernel

    def kernel_matrix(
        self,
        a: Sequence[torch.Tensor],
        b: Sequence[torch.Tensor],
        w_a: Sequence[torch.Tensor],
        w_b: Sequence[torch.Tensor],
        *,
        dtype: torch.dtype = torch.float32,
    ) -> np.ndarray:
        """Construct the Gram matrix for two collections of samples."""
        K = np.zeros((len(a), len(b)), dtype=np.float64)
        for i, xi in enumerate(a):
            for j, yj in enumerate(b):
                val = self.forward(
                    self.q_device,
                    xi.to(dtype),
                    yj.to(dtype),
                    w_a[i].to(dtype),
                    w_b[j].to(dtype),
                )
                K[i, j] = float(val.item())
        return K
