import torch
import torch.nn as nn
import torchquantum as tq
import numpy as np
from typing import Sequence
from torchquantum.functional import func_name_dict

class KernalAnsatz(tq.QuantumModule):
    """Variational quantum ansatz that encodes two feature vectors and measures overlap."""
    def __init__(self, n_wires: int) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        # Define a richer ansatz: Ry on each wire followed by a chain of CNOTs
        self.func_list = [
            {"input_idx": [i], "func": "ry", "wires": [i]} for i in range(self.n_wires)
        ] + [
            {"input_idx": None, "func": "cnot", "wires": [i, i + 1]}
            for i in range(self.n_wires - 1)
        ]

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        """Encode x and y into the device, then compute overlap."""
        q_device.reset_states(x.shape[0])
        # Encode x
        for info in self.func_list:
            params = x[:, info["input_idx"]] if info["input_idx"] is not None else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        # Reverse encode y with a sign flip
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]] if info["input_idx"] is not None else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

class Kernel(tq.QuantumModule):
    """Quantum kernel that returns the absolute overlap of two encoded states."""
    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = KernalAnsatz(n_wires=self.n_wires)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    kernel = Kernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])

class HybridKernelClassifier(tq.QuantumModule):
    """Quantum‑kernel‑based classifier that mirrors the classical variant but replaces the RBF kernel with a variational circuit."""
    def __init__(
        self,
        in_channels: int = 3,
        feature_dim: int = 120,
        num_classes: int = 2,
        n_wires: int = 4,
        support_vectors: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        # Classical feature extractor identical to the one in the classical seed
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.5),
            nn.Flatten(),
            nn.Linear(55815, 120),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
            nn.Linear(120, feature_dim),
        )
        self.kernel = Kernel(n_wires=n_wires)
        # Support vectors are learned parameters that define the kernel feature map
        if support_vectors is None:
            support_vectors = torch.randn(10, feature_dim)
        self.support_vectors = nn.Parameter(support_vectors, requires_grad=True)
        self.classifier = nn.Linear(self.support_vectors.size(0), num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        batch_size = feat.size(0)
        num_support = self.support_vectors.size(0)
        kernel_vals = torch.zeros(batch_size, num_support, device=feat.device)
        for i in range(batch_size):
            for j in range(num_support):
                kernel_vals[i, j] = self.kernel(feat[i].unsqueeze(0), self.support_vectors[j].unsqueeze(0))
        logits = self.classifier(kernel_vals)
        return logits

__all__ = ["KernalAnsatz", "Kernel", "kernel_matrix", "HybridKernelClassifier"]
