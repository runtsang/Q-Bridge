import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
from torch import nn

class KernalAnsatz(tq.QuantumModule):
    """Quantum kernel ansatz that mixes data‑encoded Ry gates with trainable Ry layers."""
    def __init__(self, n_wires: int, depth: int = 2) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.depth = depth
        # Trainable rotation angles
        self.trainable_params = nn.Parameter(torch.randn(depth, n_wires))
        # Static data‑encoding layer
        self.encoding = [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(n_wires)]

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        # Encode input x
        for info in self.encoding:
            param = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=param)
        # Apply trainable layers
        for d in range(self.depth):
            for w in range(self.n_wires):
                param = self.trainable_params[d, w]
                func_name_dict["ry"](q_device, wires=[w], params=param.expand(-1, 1))
        # Encode negative of y
        for info in reversed(self.encoding):
            param = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=param)

class Kernel(tq.QuantumModule):
    """Quantum kernel that evaluates the overlap between two data points via the ansatz."""
    def __init__(self, n_wires: int = 4, depth: int = 2) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = KernalAnsatz(self.n_wires, depth)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(-1, self.n_wires)
        y = y.reshape(-1, self.n_wires)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    """Compute the Gram matrix using the quantum ansatz."""
    kernel = Kernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])

__all__ = ["KernalAnsatz", "Kernel", "kernel_matrix"]
