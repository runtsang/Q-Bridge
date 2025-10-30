"""Hybrid classical self‑attention module that uses a quantum kernel for the attention scores.
The implementation follows the interface of the original SelfAttention class while adding a
quantum similarity measure via a TorchQuantum kernel.  The module is fully classical
(NumPy / PyTorch) and can be used as a drop‑in replacement for the seed implementation.
"""

import numpy as np
import torch
import torch.nn as nn
from torchquantum import QuantumDevice, QuantumModule
from torchquantum.functional import func_name_dict

class ClassicalSelfAttention(nn.Module):
    """Standard scaled dot‑product self‑attention."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, inputs: torch.Tensor,
                rotation_params: torch.Tensor,
                entangle_params: torch.Tensor) -> torch.Tensor:
        # Encode inputs into queries and keys via linear projections
        q = inputs @ rotation_params.reshape(self.embed_dim, -1)
        k = inputs @ entangle_params.reshape(self.embed_dim, -1)
        v = inputs
        scores = torch.softmax((q @ k.T) / np.sqrt(self.embed_dim), dim=-1)
        return scores @ v

class KernalAnsatz(QuantumModule):
    """Quantum kernel ansatz encoding two classical vectors."""
    def __init__(self, func_list):
        super().__init__()
        self.func_list = func_list

    @QuantumModule.static_support
    def forward(self, q_device: QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = x[:, info["input_idx"]] if func_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]] if func_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

class Kernel(QuantumModule):
    """Quantum kernel that returns the overlap of two encoded states."""
    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.q_device = QuantumDevice(n_wires=self.n_wires)
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

class HybridSelfAttentionKernel(nn.Module):
    """Hybrid self‑attention that uses a quantum kernel for the attention similarity."""
    def __init__(self, embed_dim: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.classical_sa = ClassicalSelfAttention(embed_dim)
        self.kernel = Kernel(n_wires=embed_dim)

    def forward(self, inputs: torch.Tensor,
                rotation_params: torch.Tensor,
                entangle_params: torch.Tensor) -> torch.Tensor:
        # Classical attention maps
        q = inputs @ rotation_params.reshape(self.embed_dim, -1)
        k = inputs @ entangle_params.reshape(self.embed_dim, -1)
        v = inputs
        # Quantum kernel similarity between each query‑key pair
        sim = torch.zeros(q.shape[0], k.shape[0])
        for i, qi in enumerate(q):
            for j, kj in enumerate(k):
                sim[i, j] = self.kernel(qi, kj)
        # Combine classical softmax with quantum similarity
        scores = torch.softmax(sim / np.sqrt(self.embed_dim), dim=-1)
        return scores @ v
