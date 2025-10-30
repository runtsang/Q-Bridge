"""Quantum hybrid kernel using a QCNN‑style ansatz and a quantum RBF kernel.

The implementation follows the structure of the seed QCNN quantum code
and the quantum kernel in QuantumKernelMethod.py.  It builds a variational
QCNN circuit, applies it to the two input feature vectors, and returns
the absolute value of the first amplitude as the kernel value.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torchquantum as tq

# --------------------------------------------------------------------------- #
# Quantum QCNN ansatz
# --------------------------------------------------------------------------- #
class _QCNNAnsatz(tq.QuantumModule):
    """Parameterised QCNN circuit used for feature encoding."""
    def __init__(self, n_wires: int) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.ansatz = tq.QuantumCircuit(n_wires)
        # Simple QCNN: layers of Ry gates followed by entangling CXs
        for i in range(n_wires):
            self.ansatz.ry(i, params=f"theta_{i}")
        for i in range(n_wires - 1):
            self.ansatz.cx(i, i + 1)

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor) -> None:
        """Encode input vector `x` into the circuit."""
        for i in range(self.n_wires):
            param = x[:, i:i + 1] if x.shape[1] > i else None
            tq.ry(q_device, wires=[i], params=param)

# --------------------------------------------------------------------------- #
# Quantum RBF kernel (state‑overlap based)
# --------------------------------------------------------------------------- #
class _QuantumRBFKernel(tq.QuantumModule):
    """Quantum kernel that measures overlap between two encoded states."""
    def __init__(self, n_wires: int) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        """Apply the encoding for `x` and then for `-y`."""
        q_device.reset_states(x.shape[0])
        # Encode x
        for i in range(self.n_wires):
            param = x[:, i:i + 1] if x.shape[1] > i else None
            tq.ry(q_device, wires=[i], params=param)
        # Encode -y
        for i in range(self.n_wires):
            param = -y[:, i:i + 1] if y.shape[1] > i else None
            tq.ry(q_device, wires=[i], params=param)

    def kernel_value(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self.forward(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

# --------------------------------------------------------------------------- #
# Hybrid quantum kernel combining QCNN ansatz and RBF evaluation
# --------------------------------------------------------------------------- #
class HybridQuantumKernelCNN(tq.QuantumModule):
    """Hybrid quantum kernel: QCNN feature encoding followed by state‑overlap."""
    def __init__(self, n_wires: int = 8) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.qcnn = _QCNNAnsatz(n_wires)
        self.kernel = _QuantumRBFKernel(n_wires)

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        """Compute kernel value between `x` and `y`."""
        self.kernel.forward(q_device, x, y)

    def kernel_value(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.kernel.kernel_value(x, y)

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        return np.array([[self.kernel_value(torch.tensor(x), torch.tensor(y)).item() for y in b] for x in a])

__all__ = ["HybridQuantumKernelCNN"]
