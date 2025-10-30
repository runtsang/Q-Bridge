import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
from torch.nn import Parameter
from typing import Sequence

class QuantumKernelMethod(tq.QuantumModule):
    """Quantum kernel implemented with TorchQuantum; uses a trainable Ry ansatz."""
    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        # Trainable parameters for rotation angles
        self.params = Parameter(torch.randn(n_wires))
        # Build a simple ansatz: Ry gates on each qubit
        self.ansatz = [
            {"input_idx": [i], "func": "ry", "wires": [i]} for i in range(n_wires)
        ]
    @tq.static_support
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Reset device with batch size from x
        self.q_device.reset_states(x.shape[0])
        # Encode x
        for info in self.ansatz:
            idx = info["input_idx"][0]
            func = info["func"]
            wires = info["wires"]
            params = x[:, idx] if tq.op_name_dict[func].num_params else None
            func_name_dict[func](self.q_device, wires=wires, params=params)
        # Apply inverse encoding with y (negative angles)
        for info in reversed(self.ansatz):
            idx = info["input_idx"][0]
            func = info["func"]
            wires = info["wires"]
            params = -y[:, idx] if tq.op_name_dict[func].num_params else None
            func_name_dict[func](self.q_device, wires=wires, params=params)
        # Return the absolute amplitude of |0...0> as kernel value
        return torch.abs(self.q_device.states.view(-1)[0])
    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        kernel = torch.zeros((len(a), len(b)))
        for i, xi in enumerate(a):
            for j, yj in enumerate(b):
                kernel[i, j] = self.forward(xi, yj)
        return kernel.cpu().numpy()

__all__ = ["QuantumKernelMethod"]
