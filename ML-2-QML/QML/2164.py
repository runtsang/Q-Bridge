import numpy as np
import torch
from torch import nn
import torchquantum as tq
from torchquantum.functional import func_name_dict
from typing import Sequence

class QuantumAnsatz(tq.QuantumModule):
    """Parameterized ansatz with trainable Ry rotations and CNOT entanglement."""

    def __init__(self, n_wires: int, n_layers: int):
        super().__init__()
        self.n_wires = n_wires
        self.n_layers = n_layers
        self.params = nn.Parameter(torch.randn(n_layers, n_wires))

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        # Encode x
        for i in range(self.n_wires):
            func_name_dict["ry"](q_device, wires=i, params=x[:, i])
        # Trainable layers
        for layer in range(self.n_layers):
            for i in range(self.n_wires - 1):
                func_name_dict["cx"](q_device, wires=[i, i + 1])
            for i in range(self.n_wires):
                func_name_dict["ry"](q_device, wires=i, params=self.params[layer, i])
        # Encode y with negative phases to implement overlap
        for i in range(self.n_wires):
            func_name_dict["ry"](q_device, wires=i, params=-y[:, i])

class HybridKernel(tq.QuantumModule):
    """Hybrid kernel combining classical RBF, polynomial, linear and a trainable quantum overlap."""

    def __init__(
        self,
        n_wires: int = 4,
        n_layers: int = 2,
        gamma: float = 1.0,
        poly_degree: int = 3,
        weight: Sequence[float] | None = None,
    ):
        super().__init__()
        self.n_wires = n_wires
        self.n_layers = n_layers
        self.gamma = gamma
        self.poly_degree = poly_degree
        self.ansatz = QuantumAnsatz(n_wires, n_layers)
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        if weight is None:
            weight = [1.0, 1.0, 1.0, 0.0]  # RBF, poly, linear, quantum
        self.weight = torch.tensor(weight, dtype=torch.float32)

    def _rbf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x.unsqueeze(1) - y.unsqueeze(0)
        return torch.exp(-self.gamma * torch.sum(diff ** 2, dim=-1))

    def _poly(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        dot = torch.matmul(x, y.t())
        return (dot + 1.0) ** self.poly_degree

    def _linear(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, y.t())

    def _quantum_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        k_rbf = self._rbf(x, y)
        k_poly = self._poly(x, y)
        k_lin = self._linear(x, y)
        k_q = self._quantum_kernel(x, y)
        return (
            self.weight[0] * k_rbf
            + self.weight[1] * k_poly
            + self.weight[2] * k_lin
            + self.weight[3] * k_q
        )

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        x = torch.stack(a)
        y = torch.stack(b)
        return self.forward(x, y).detach().cpu().numpy()

__all__ = ["HybridKernel"]
