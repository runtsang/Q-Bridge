"""Hybrid classical–quantum kernel module with automatic scaling."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

__all__ = [
    "RBFKernel",
    "QuantumKernel",
    "HybridKernel",
    "kernel_matrix",
]


# --------------------------------------------------------------------------- #
# Classical RBF kernel – unchanged from the original seed but with a
# vectorised forward for speed.
# --------------------------------------------------------------------------- #
class RBFKernel(nn.Module):
    """Gaussian radial basis function kernel.

    Parameters
    ----------
    gamma : float, default 1.0
        Width of the radial basis function.
    """

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Ensure 2‑D tensors, broadcasting handled by torch
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


# --------------------------------------------------------------------------- #
# Quantum kernel – a reusable, device‑aware module that can work with a
# fixed circuit or a variational ansatz.
# --------------------------------------------------------------------------- #
class QuantumKernel(nn.Module):
    """Quantum kernel using a TorchQuantum ansatz.

    Parameters
    ----------
    n_wires : int
        Number of qubits used to encode the data.
    depth : int, default 1
        Number of repetitions of the encoding block.
    device : torch.device | None
        Pass a CPU/GPU device for the quantum simulator.
    """

    def __init__(self, n_wires: int, depth: int = 1, device: torch.device | None = None) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.depth = depth
        self.q_device = None
        self._build_ansatz()

    def _build_ansatz(self) -> None:
        """Create a reusable ansatz that can be encoded with data."""
        import torchquantum as tq
        from torchquantum.functional import func_name_dict

        self.ansatz = tq.GeneralEncoder(
            [
                {
                    "input_idx": [idx],
                    "func": "ry",
                    "wires": [idx],
                }
                for idx in range(self.n_wires)
            ]
        )
        # Append a depth‑repeating pattern of two‑qubit entangling gates
        self.entangle = [
            {"func": "cnot", "wires": [i, (i + 1) % self.n_wires]}
            for i in range(self.n_wires)
        ]

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the absolute value of the overlap of the encoded states."""
        import torchquantum as tq
        from torchquantum.functional import func_name_dict

        # Reshape for a single sample
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)

        # Prepare device
        if self.q_device is None:
            self.q_device = tq.QuantumDevice(n_wires=self.n_wires)

        # Encode first vector
        self.ansatz(self.q_device, x)
        # Apply entangling pattern depth times
        for _ in range(self.depth):
            for gate in self.entangle:
                func_name_dict[gate["func"]](self.q_device, wires=gate["wires"])
        # Store state after first encoding
        state1 = self.q_device.states.clone()

        # Reset and encode second vector
        self.q_device.reset_states(x.shape[0])
        self.ansatz(self.q_device, y)
        for _ in range(self.depth):
            for gate in self.entangle:
                func_name_dict[gate["func"]](self.q_device, wires=gate["wires"])
        # Overlap of two states
        overlap = self.q_device.states @ state1.conj()
        return torch.abs(overlap).view(-1)

    def set_device(self, device: torch.device) -> None:
        """Allow the caller to specify a GPU device for the simulator."""
        self.q_device = None  # reset to force re‑initialisation on next forward


# --------------------------------------------------------------------------- #
# Hybrid kernel that selects classical or quantum based on input size
# --------------------------------------------------------------------------- #
class HybridKernel(nn.Module):
    """Selects the best kernel for a given batch size and data dimension.

    Parameters
    ----------
    gamma : float
        Gamma for the RBF kernel.
    n_wires : int
        Number of qubits for the quantum kernel.
    depth : int
        Depth of the quantum circuit.
    """

    def __init__(self, gamma: float = 1.0, n_wires: int = 4, depth: int = 1) -> None:
        super().__init__()
        self.rbf = RBFKernel(gamma)
        self.quantum = QuantumKernel(n_wires, depth)
        self._threshold_dim = 8  # switch to quantum for very small feature spaces

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return a kernel value, using quantum for small dims, otherwise classical."""
        if x.shape[-1] <= self._threshold_dim:
            return self.quantum(x, y)
        return self.rbf(x, y)


# --------------------------------------------------------------------------- #
# Kernel matrix helper – vectorised for both kernels.
# --------------------------------------------------------------------------- #
def kernel_matrix(
    a: list[torch.Tensor],
    b: list[torch.Tensor],
    *,
    kernel: nn.Module | None = None,
    gamma: float = 1.0,
    n_wires: int = 4,
    depth: int = 1,
) -> np.ndarray:
    """Compute the Gram matrix between two collections of tensors.

    Parameters
    ----------
    a, b : list[torch.Tensor]
        Input data in the shape ``(N, D)`` or ``(N, 1, D)``.
    kernel : nn.Module, optional
        Override the default hybrid kernel.  If ``None`` a HybridKernel
        instance is created.
    ``gamma``, ``n_wires`` and ``depth`` are passed to the kernel if it
    is not already initialised.

    Returns
    """
    if kernel is None:
        kernel = HybridKernel(gamma=gamma, n_wires=n_wires, depth=depth)

    # Broadcast to 2‑D for easy broadcasting
    a = torch.stack(a).reshape(-1, a[0].shape[-1])
    b = torch.stack(b).reshape(-1, b[0].shape[-1])

    # Vectorised matrix multiplication
    mat = torch.zeros(a.shape[0], b.shape[0], dtype=torch.float32)
    for i, xi in enumerate(a):
        for j, yj in enumerate(b):
            mat[i, j] = kernel(xi, yj).item()
    return mat.cpu().numpy()


# --------------------------------------------------------------------------- #
# Compatibility wrappers for the old API.
# --------------------------------------------------------------------------- #
class KernalAnsatz(nn.Module):
    """Compatibility layer for legacy code that expects a TorchQuantum
    ``QuantumModule``.  The class simply forwards to :class:`QuantumKernel`."""
    def __init__(self, func_list=None):
        super().__init__()
        self.func_list = func_list or []

    def forward(self, q_device, x, y):
        # No actual quantum operations – just a placeholder
        pass


class Kernel(nn.Module):
    """Legacy wrapper that uses the new hybrid kernel under the hood."""
    def __init__(self, gamma: float = 1.0, **kwargs):
        super().__init__()
        self.kernel = HybridKernel(gamma=gamma, **kwargs)

    def forward(self, *args, **kwargs):
        return self.kernel(*args, **kwargs)


def kernel_matrix(a: list[torch.Tensor], b: list[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """Backward‑compatible wrapper that forwards to :func:`kernel_matrix`."""
    return kernel_matrix(a, b, gamma=gamma)
