"""Hybrid quantum–classical kernel method.

The :class:`HybridKernelMethod` class integrates a TorchQuantum ansatz with
a quantum convolution filter.  It can be used as a drop‑in replacement for
the legacy ``QuantumKernelMethod`` while providing a richer feature
extraction pipeline.
"""

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict


class QuantumConvFilter(tq.QuantumModule):
    """Quantum analogue of a 2‑D convolution filter using a single qubit."""
    def __init__(self, threshold: float = 0.127):
        super().__init__()
        self.threshold = threshold
        self.device = tq.QuantumDevice(n_wires=1)

    @tq.static_support
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Encode the mean of each data vector into a rotation and return
        the probability of measuring |1>.
        """
        batch = data.shape[0]
        self.device.reset_states(batch)
        for i in range(batch):
            mean_val = data[i].mean()
            angle = torch.where(mean_val > self.threshold,
                                torch.tensor(np.pi),
                                torch.tensor(0.0))
            tq.ry(self.device, wires=0, params=angle)
        probs = self.device.get_probs()
        return probs[:, 1]  # probability of |1>


class QuantumAnsatz(tq.QuantumModule):
    """Fixed ansatz that estimates the overlap between two classical vectors."""
    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.device = tq.QuantumDevice(n_wires=self.n_wires)

    @tq.static_support
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the fidelity between |x⟩ and |y⟩ using a simple circuit."""
        batch = x.shape[0]
        self.device.reset_states(batch)
        # Encode x on the first half of wires
        for i in range(self.n_wires // 2):
            tq.ry(self.device, wires=i, params=x[:, i])
        # Encode y on the second half
        for i in range(self.n_wires // 2):
            tq.ry(self.device, wires=self.n_wires // 2 + i,
                  params=y[:, i])
        # Simple entanglement
        tq.cnot(self.device, wires=[0, 1])
        tq.cnot(self.device, wires=[2, 3])
        probs = self.device.get_probs()
        return probs[:, 0]  # probability of all‑zero state


class HybridKernelMethod(tq.QuantumModule):
    """Hybrid kernel that combines a quantum convolution filter with a fixed ansatz."""
    def __init__(self,
                 n_wires: int = 4,
                 conv_threshold: float = 0.127):
        super().__init__()
        self.ansatz = QuantumAnsatz(n_wires)
        self.conv_filter = QuantumConvFilter(conv_threshold)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the hybrid kernel value between two vectors."""
        # Pre‑processing via quantum convolution
        x_conv = self.conv_filter(x).unsqueeze(-1)  # shape (batch, 1)
        y_conv = self.conv_filter(y).unsqueeze(-1)
        # Scale the ansatz output by the convolution probabilities
        ansatz_val = self.ansatz(x, y)
        return ansatz_val * x_conv * y_conv

    def kernel_matrix(self,
                      a: Sequence[torch.Tensor],
                      b: Sequence[torch.Tensor]) -> np.ndarray:
        """Compute a Gram matrix between two collections of vectors."""
        kernel_vals: List[List[float]] = []
        for x in a:
            row: List[float] = []
            for y in b:
                val = self.forward(x.unsqueeze(0), y.unsqueeze(0))
                row.append(val.item())
            kernel_vals.append(row)
        return np.array(kernel_vals)


__all__ = ["HybridKernelMethod"]
