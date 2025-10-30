"""Quantum kernel with depth‑controlled entanglement and learnable parameters."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict


# --------------------------------------------------------------------------- #
#  Parameterised feature map
# --------------------------------------------------------------------------- #
class _FeatureMap(tq.QuantumModule):
    """Depth‑controlled, entangling feature map."""
    def __init__(self,
                 n_wires: int,
                 depth: int = 2,
                 entanglement: str = "full") -> None:
        super().__init__()
        self.n_wires = n_wires
        self.depth = depth
        self.entanglement = entanglement
        self.params = nn.Parameter(torch.randn(depth, n_wires))

    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for d in range(self.depth):
            # Rotation layer
            for w in range(self.n_wires):
                func_name_dict["ry"](q_device, wires=[w], params=x[:, w] * self.params[d, w])
            # Entanglement layer
            if self.entanglement == "full":
                for w in range(self.n_wires):
                    func_name_dict["cx"](q_device, wires=[w, (w + 1) % self.n_wires])
            elif self.entanglement == "linear":
                for w in range(self.n_wires - 1):
                    func_name_dict["cx"](q_device, wires=[w, w + 1])


# --------------------------------------------------------------------------- #
#  Kernel module
# --------------------------------------------------------------------------- #
class KernalAnsatz(tq.QuantumModule):
    """Quantum kernel that applies a feature map and measures overlap."""
    def __init__(self,
                 n_wires: int = 4,
                 depth: int = 2,
                 entanglement: str = "full") -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.feature_map = _FeatureMap(self.n_wires, depth, entanglement)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Forward x
        self.feature_map(self.q_device, x)
        state_x = self.q_device.states.clone()
        # Forward y with negative parameters to compute overlap
        self.feature_map(self.q_device, -y)
        state_y = self.q_device.states
        # Overlap (real part)
        overlap = torch.abs(torch.sum(state_x * state_y.conj(), dim=-1, keepdim=True))
        return overlap.squeeze()


# --------------------------------------------------------------------------- #
#  Wrapper with optional trainability
# --------------------------------------------------------------------------- #
class Kernel(tq.QuantumModule):
    """Convenience wrapper that exposes depth and trainable flag."""
    def __init__(self,
                 n_wires: int = 4,
                 depth: int = 2,
                 entanglement: str = "full",
                 trainable: bool = True) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(n_wires, depth, entanglement)
        self.trainable = trainable
        if not trainable:
            for p in self.ansatz.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.ansatz(x, y)


# --------------------------------------------------------------------------- #
#  Gram matrix utility
# --------------------------------------------------------------------------- #
def kernel_matrix(a: Sequence[torch.Tensor],
                  b: Sequence[torch.Tensor],
                  n_wires: int = 4,
                  depth: int = 2,
                  entanglement: str = "full",
                  trainable: bool = True) -> np.ndarray:
    kernel = Kernel(n_wires, depth, entanglement, trainable)
    return np.array([[kernel(x, y).item() for y in b] for x in a])


__all__ = ["_FeatureMap", "KernalAnsatz", "Kernel", "kernel_matrix"]
