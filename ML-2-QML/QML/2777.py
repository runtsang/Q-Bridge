import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict

class QuantumKernel(tq.QuantumModule):
    """
    A lightweight quantum kernel that encodes two classical feature vectors
    using Ry rotations, applies a shared variational ansatz, and measures
    the absolute overlap of the resulting states.
    """
    def __init__(self, n_wires: int = 8):
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=n_wires)
        self.params = torch.nn.Parameter(torch.randn(n_wires, requires_grad=True))

    def _encode(self, x: torch.Tensor, sign: int = 1):
        for i in range(self.n_wires):
            func_name_dict["ry"](self.q_device, wires=i, params=sign * x[:, i])

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        self.q_device.reset_states(batch_size)
        self._encode(x, sign=1)
        self._encode(y, sign=-1)
        for i in range(self.n_wires):
            func_name_dict["ry"](self.q_device, wires=i, params=self.params[i])
        for i in range(self.n_wires - 1):
            func_name_dict["cx"](self.q_device, wires=[i, i + 1])
        return torch.abs(self.q_device.states.view(-1)[0])

    def kernel_matrix(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [self.forward(a[i:i+1], b) for i in range(a.size(0))]
        ).squeeze()

__all__ = ["QuantumKernel"]
