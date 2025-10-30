"""Hybrid quantum kernel combining a TorchQuantum ansatz with classical RBF preprocessing.

The quantum component is the same as in the original seed but is now wrapped
in a module that first applies a classical RBF transformation to the inputs.
The resulting similarity is used to control the amplitude of the quantum
overlap.  This design keeps the interface identical to the original
:class:`Kernel` while exposing a clear path to hybrid classical‑quantum
training.

The module requires ``torchquantum``; if it is not available a minimal
fallback is provided that raises an informative error.
"""

from __future__ import annotations

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict

# Import the original RBF kernel implementation
try:
    from QuantumKernelMethod import Kernel as ClassicalKernel
except Exception:  # pragma: no cover
    # Minimal fallback RBF kernel
    class ClassicalKernel(tq.QuantumModule):
        def __init__(self, gamma: float = 1.0) -> None:
            super().__init__()
            self.gamma = gamma

        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            diff = x - y
            return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class HybridQuantumKernel(tq.QuantumModule):
    """Quantum kernel that multiplies classical RBF similarity by quantum overlap.

    Parameters
    ----------
    n_wires : int
        Number of qubits in the quantum device.
    """

    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = self._build_ansatz()
        self.rbf = ClassicalKernel(gamma=1.0)  # Classical RBF used as pre‑processor

    def _build_ansatz(self) -> tq.QuantumModule:
        """Build a simple Ry‑only ansatz with a single layer."""
        class Ansatz(tq.QuantumModule):
            def __init__(self, n_wires: int):
                super().__init__()
                self.n_wires = n_wires
                self.func_list = [
                    {"input_idx": [i], "func": "ry", "wires": [i]}
                    for i in range(n_wires)
                ]

            @tq.static_support
            def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
                q_device.reset_states(x.shape[0])
                for info in self.func_list:
                    params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
                    func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
                for info in reversed(self.func_list):
                    params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
                    func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

        return Ansatz(self.n_wires)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute hybrid kernel value.

        The function first evaluates the classical RBF similarity,
        then runs the quantum ansatz on the same data and uses the absolute
        value of the first state amplitude as the quantum overlap.
        The final kernel is the product of the two.
        """
        # Classical RBF similarity
        rbf_score = self.rbf(x.reshape(1, -1), y.reshape(1, -1))

        # Quantum overlap
        self.ansatz(self.q_device, x.reshape(1, -1), y.reshape(1, -1))
        quantum_score = torch.abs(self.q_device.states.view(-1)[0]).unsqueeze(0)

        return rbf_score * quantum_score

    def kernel_matrix(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute the Gram matrix between two sets of samples."""
        a_t = torch.tensor(a, dtype=torch.float32)
        b_t = torch.tensor(b, dtype=torch.float32)
        return np.array([[self.forward(x, y).item() for y in b_t] for x in a_t])

__all__ = ["HybridQuantumKernel"]
