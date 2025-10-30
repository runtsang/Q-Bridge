"""Hybrid classical kernel + QCNN implementation using PyTorch and TorchQuantum."""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import numpy as np
import torchquantum as tq
from torchquantum.functional import func_name_dict


class ClassicalRBFKernel(nn.Module):
    """Exponentially decaying kernel with learnable gamma."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x, y are 1‑D tensors of shape (D,)
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff))


class QCNNModel(nn.Module):
    """Fully‑connected network mimicking a QCNN."""
    def __init__(self) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))


class QuantumKernel(tq.QuantumModule):
    """Fixed TorchQuantum ansatz for kernel evaluation."""
    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = KernalAnsatz([
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


class KernalAnsatz(tq.QuantumModule):
    """Encodes two classical vectors into a quantum device."""
    def __init__(self, func_list: Sequence[dict]) -> None:
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


class HybridKernelQCNN(nn.Module):
    """Hybrid module combining a QCNN feature extractor with a kernel."""
    def __init__(self, gamma: float = 1.0, n_wires: int = 4) -> None:
        super().__init__()
        self.qcnn = QCNNModel()
        self.kernel = ClassicalRBFKernel(gamma)
        self.qkernel = QuantumKernel(n_wires)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return QCNN output as a feature vector."""
        return self.qcnn(x).detach()

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """Compute Gram matrix using QCNN features and classical RBF kernel."""
        feats_a = torch.stack([self.extract_features(x) for x in a])
        feats_b = torch.stack([self.extract_features(y) for y in b])
        return np.array([[self.kernel(x, y).item() for y in feats_b] for x in feats_a])

    def quantum_kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """Compute Gram matrix using quantum kernel on QCNN features."""
        feats_a = torch.stack([self.extract_features(x) for x in a])
        feats_b = torch.stack([self.extract_features(y) for y in b])
        return np.array([[self.qkernel(x, y).item() for y in feats_b] for x in feats_a])


__all__ = ["HybridKernelQCNN", "ClassicalRBFKernel", "QCNNModel", "QuantumKernel", "KernalAnsatz"]
