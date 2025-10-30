import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
from typing import Sequence

class QuantumAnsatz(tq.QuantumModule):
    """Variational ansatz for quantum kernel."""
    def __init__(self, param_list):
        super().__init__()
        self.param_list = param_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.param_list:
            params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.param_list):
            params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

class QuantumSampler(tq.QuantumModule):
    """Quantum sampler network analogous to classical SamplerQNN."""
    def __init__(self, n_wires: int = 2):
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.circuit = [
            {"input_idx": [0], "func": "ry", "wires": [0]},
            {"input_idx": [1], "func": "ry", "wires": [1]},
            {"func": "cx", "wires": [0, 1]},
            {"input_idx": [0], "func": "ry", "wires": [0]},
            {"input_idx": [1], "func": "ry", "wires": [1]},
        ]

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.circuit:
            params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

    def sample(self, inputs: torch.Tensor) -> torch.Tensor:
        self.forward(self.q_device, inputs)
        return self.q_device.states.view(-1)[0].abs()

class HybridKernelMethod(tq.QuantumModule):
    """Hybrid quantum kernel that can be used standalone or as a component in a larger model."""
    def __init__(self, n_wires: int = 4, param_list: list | None = None):
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        if param_list is None:
            param_list = [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        self.ansatz = QuantumAnsatz(param_list)
        self.sampler = QuantumSampler(n_wires=self.n_wires)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        return np.array([[self.forward(x, y).item() for y in b] for x in a])

    def sample(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.sampler.sample(inputs)

__all__ = ["QuantumAnsatz", "QuantumSampler", "HybridKernelMethod"]
