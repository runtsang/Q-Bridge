"""Quantum kernel construction using TorchQuantum ansatz with normalisation."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict


class QuantumKernelMethod(tq.QuantumModule):
    """Quantum kernel evaluated via a user‑defined ansatz.

    Parameters
    ----------
    ansatz : Sequence[dict] | None, default None
        List of dictionaries describing each gate to apply.
    n_wires : int, default 4
        Number of qubits.
    normalize : bool, default False
        Normalise the resulting Gram matrix.
    """
    def __init__(
        self,
        ansatz: Sequence[dict] | None = None,
        n_wires: int = 4,
        normalize: bool = False,
    ) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=n_wires)
        self.normalize = normalize
        self.ansatz = ansatz or [
            {"input_idx": [0], "func": "ry", "wires": [0]},
            {"input_idx": [1], "func": "ry", "wires": [1]},
            {"input_idx": [2], "func": "ry", "wires": [2]},
            {"input_idx": [3], "func": "ry", "wires": [3]},
        ]

    @tq.static_support
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the kernel value for two single‑example vectors."""
        self.q_device.reset_states(1)
        for info in self.ansatz:
            params = x[info["input_idx"]]
            func_name_dict[info["func"]](self.q_device,
                                         wires=info["wires"],
                                         params=params)
        for info in reversed(self.ansatz):
            params = -y[info["input_idx"]]
            func_name_dict[info["func"]](self.q_device,
                                         wires=info["wires"],
                                         params=params)
        return torch.abs(self.q_device.states.view(-1)[0])

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        a = [torch.as_tensor(v, dtype=torch.float32) for v in a]
        b = [torch.as_tensor(v, dtype=torch.float32) for v in b]
        K = np.array([[self.forward(x, y).item() for y in b] for x in a])
        if self.normalize:
            Kaa = np.array([[self.forward(x, y).item() for y in a] for x in a])
            Kbb = np.array([[self.forward(x, y).item() for y in b] for x in b])
            diag_a = np.diag(Kaa)
            diag_b = np.diag(Kbb)
            denom = np.sqrt(np.outer(diag_a, diag_b))
            K = K / denom
        return K


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor],
                  ansatz: Sequence[dict] | None = None,
                  n_wires: int = 4,
                  normalize: bool = False) -> np.ndarray:
    """Convenience wrapper that returns the kernel matrix for two datasets."""
    qkm = QuantumKernelMethod(ansatz=ansatz,
                              n_wires=n_wires,
                              normalize=normalize)
    return qkm.kernel_matrix(a, b)


__all__ = ["QuantumKernelMethod", "kernel_matrix"]
