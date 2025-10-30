"""Quantum hybrid implementation of the binary classifier.

This module builds upon the classical backbone but replaces the kernel head
with a parameterised quantum kernel evaluated via TorchQuantum. The
quantum kernel captures non‑linear feature interactions beyond the reach of
a classical RBF, while the rest of the network remains identical to the
classical counterpart.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
from torchquantum.functional import func_name_dict

class QuantumKernAnsatz(tq.QuantumModule):
    """Parameterized ansatz that encodes two classical vectors."""
    def __init__(self, gate_specs):
        super().__init__()
        self.gate_specs = gate_specs

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for spec in self.gate_specs:
            params = x[:, spec["input_idx"]] if tq.op_name_dict[spec["func"]].num_params else None
            func_name_dict[spec["func"]](q_device, wires=spec["wires"], params=params)
        for spec in reversed(self.gate_specs):
            params = -y[:, spec["input_idx"]] if tq.op_name_dict[spec["func"]].num_params else None
            func_name_dict[spec["func"]](q_device, wires=spec["wires"], params=params)

class QuantumKernel(tq.QuantumModule):
    """Quantum kernel computed via the ansatz."""
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = QuantumKernAnsatz(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0]).unsqueeze(-1)

class HybridQuantumKernelNet(nn.Module):
    """Hybrid network with quantum kernel head."""
    def __init__(self, num_prototypes: int = 10) -> None:
        super().__init__()
        # Convolutional backbone identical to the classical version
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        # Quantum kernel head
        self.quantum_kernel = QuantumKernel()
        # Prototypes for kernel evaluation
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, 1))
        self.kernel_head = nn.Linear(num_prototypes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # (batch, 1)
        # Compute quantum kernel similarities
        sims = [self.quantum_kernel(x, p.unsqueeze(0)) for p in self.prototypes]
        kernel_sim = torch.cat(sims, dim=-1)  # (batch, num_prototypes)
        logits = self.kernel_head(kernel_sim)
        probs = torch.sigmoid(logits)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["HybridQuantumKernelNet"]
