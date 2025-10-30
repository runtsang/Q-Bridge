import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchquantum as tq
from torchquantum.functional import func_name_dict
from typing import Sequence

class RBFAnsatz(tq.QuantumModule):
    """Quantum RBF kernel ansatz that encodes two vectors with Ry gates."""
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

class RBFKernel(tq.QuantumModule):
    """Quantum kernel that evaluates overlap between two encoded states."""
    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = RBFAnsatz([
            {"input_idx": [0], "func": "ry", "wires": [0]},
            {"input_idx": [1], "func": "ry", "wires": [1]},
            {"input_idx": [2], "func": "ry", "wires": [2]},
            {"input_idx": [3], "func": "ry", "wires": [3]},
        ])
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

class HybridQuanvolutionFilter(tq.QuantumModule):
    """Quantum filter that applies a quantum kernel to each 2×2 patch."""
    def __init__(self, patch_size: int = 2, stride: int = 2, in_channels: int = 1, n_wires: int = 4):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.in_channels = in_channels
        self.n_wires = n_wires
        self.kernel = RBFKernel(n_wires=self.n_wires)
        self.center = torch.rand(1, n_wires)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        patches = []
        for r in range(0, 28, self.stride):
            for c in range(0, 28, self.stride):
                patch = x[:, :, r:r+self.patch_size, c:c+self.patch_size]
                patch = patch.view(bsz, -1)
                k = self.kernel(patch, self.center)
                patches.append(k)
        return torch.cat(patches, dim=1).unsqueeze(1)  # (B, 1, num_patches)

class HybridQuanvolution(tq.QuantumModule):
    """Hybrid classifier that uses the quantum quanvolution filter followed by a classical linear head."""
    def __init__(self, num_classes: int = 10, n_wires: int = 4):
        super().__init__()
        self.filter = HybridQuanvolutionFilter(n_wires=n_wires)
        self.linear = nn.Linear(self.filter.kernel.n_wires, num_classes)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.filter(x)
        logits = self.linear(feats.view(feats.size(0), -1))
        return F.log_softmax(logits, dim=-1)

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    kernel = RBFKernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])

__all__ = ["HybridQuanvolution", "kernel_matrix"]
