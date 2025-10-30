import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
import numpy as np

class KernalAnsatz(tq.QuantumModule):
    """
    Encodes classical data through a programmable list of quantum gates.
    """
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

class QuantumHybridKernelRegressor(tq.QuantumModule):
    """
    Quantum kernel evaluator that can be injected into the hybrid regressor.
    Supports a configurable layered ansatz.
    """
    def __init__(self, n_wires: int = 4, n_layers: int = 2):
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = self._build_ansatz(n_wires, n_layers)

    def _build_ansatz(self, n_wires, n_layers):
        func_list = []
        for layer in range(n_layers):
            for w in range(n_wires):
                func_list.append({"input_idx": [layer * n_wires + w], "func": "ry", "wires": [w]})
        return KernalAnsatz(func_list)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

__all__ = ["KernalAnsatz", "QuantumHybridKernelRegressor"]
