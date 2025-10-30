import numpy as np
import torch
import torchquantum as tq
from torch import nn
from typing import Sequence

class QuantumKernelMethod__gen126(tq.QuantumModule):
    """
    Variational quantum kernel with trainable rotation weights.
    Implements a fixed ansatz: Ry and Rz on each qubit followed by a CNOT chain.
    The kernel value is the absolute overlap between the states prepared from x and y.
    """
    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        # Trainable weights for Ry and Rz
        self.weight_Ry = nn.Parameter(torch.randn(self.n_wires))
        self.weight_Rz = nn.Parameter(torch.randn(self.n_wires))
        # Build static ansatz list
        self.ansatz = self._build_ansatz()

    def _build_ansatz(self):
        ansatz = []
        for i in range(self.n_wires):
            ansatz.append({"func": "ry", "wires": [i]})
            ansatz.append({"func": "rz", "wires": [i]})
        for i in range(self.n_wires - 1):
            ansatz.append({"func": "cx", "wires": [i, i + 1]})
        return ansatz

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        # Reset device states
        q_device.reset_states(x.shape[0])
        # Encode x
        for i, gate in enumerate(self.ansatz):
            if gate["func"] == "ry":
                angle = x[:, i] * self.weight_Ry[i]
                tq.functional.ry(q_device, wires=[i], params=angle)
            elif gate["func"] == "rz":
                angle = x[:, i] * self.weight_Rz[i]
                tq.functional.rz(q_device, wires=[i], params=angle)
            elif gate["func"] == "cx":
                tq.functional.cx(q_device, wires=gate["wires"])
        # Encode y with negative sign
        for i, gate in reversed(list(enumerate(self.ansatz))):
            if gate["func"] == "ry":
                angle = -y[:, i] * self.weight_Ry[i]
                tq.functional.ry(q_device, wires=[i], params=angle)
            elif gate["func"] == "rz":
                angle = -y[:, i] * self.weight_Rz[i]
                tq.functional.rz(q_device, wires=[i], params=angle)
            elif gate["func"] == "cx":
                tq.functional.cx(q_device, wires=gate["wires"])

    def kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute kernel value for two input vectors.
        """
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.forward(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """
        Compute Gram matrix between two sequences of tensors.
        """
        a = torch.stack(a)
        b = torch.stack(b)
        results = []
        for x in a:
            row = []
            for y in b:
                val = self.kernel(x, y)
                row.append(val.item())
            results.append(row)
        return np.array(results)

__all__ = ["QuantumKernelMethod__gen126"]
