"""Hybrid classical‑quantum kernel module providing GPU‑accelerated RBF, a learnable variational quantum kernel, and a selector.

The module keeps backward compatibility with the original seed by exposing the names
``KernalAnsatz`` and ``Kernel``.  New classes are added:
* :class:`RBFKernel` – fast GPU implementation of the radial basis function.
* :class:`VariationalAnsatz` – a parameterised circuit that can be trained.
* :class:`QuantumKernel` – evaluates the quantum kernel using :class:`VariationalAnsatz`.
* :class:`KernelSelector` – runtime switch between classical and quantum kernels.
* :class:`PostProcessor` – lightweight neural net that maps a kernel matrix to a scalar output.
* :func:`kernel_matrix` – convenience wrapper that chooses the appropriate kernel.

These extensions enable end‑to‑end training of a kernel‑based regression model while still
allowing a classical baseline for comparison.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from typing import Sequence, Callable, Optional

# --- 1. Classical RBF kernel -------------------------------------------------
class RBFKernel(nn.Module):
    """GPU‑accelerated RBF kernel.

    Parameters
    ----------
    gamma : float, optional
        Exponential decay rate.  Default is 1.0.
    device : torch.device, optional
        Device on which the kernel will be evaluated.  If ``None``, the
        default device of the input tensors is used.
    """
    def __init__(self, gamma: float = 1.0, device: torch.device | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.device = device

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute ``exp(-γ‖x−y‖²)`` for two 2‑D tensors ``x`` and ``y``.

        The tensors are expected to be of shape ``(N, D)`` and ``(M, D)``.
        """
        x = x.to(self.device)
        y = y.to(self.device)
        # Using broadcasting to compute pairwise squared distances
        diff = x.unsqueeze(1) - y.unsqueeze(0)  # (N, M, D)
        dist_sq = torch.sum(diff * diff, dim=2)  # (N, M)
        return torch.exp(-self.gamma * dist_sq)

# --- 2. Classical ansatz (back‑compat) ---------------------------------------
class KernalAnsatz(nn.Module):
    """Legacy RBF ansatz that keeps the original API."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    """Legacy wrapper around :class:`KernalAnsatz`."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

# --- 3. Variational quantum kernel ------------------------------------------
import torchquantum as tq
from torchquantum.functional import func_name_dict

class VariationalAnsatz(tq.QuantumModule):
    """Parameterised circuit that learns optimal encoding.

    The circuit consists of ``depth`` layers of single‑qubit rotations followed
    by a layer of CZ gates.  All rotation angles are learnable parameters.
    """
    def __init__(self, n_wires: int, depth: int = 2) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.depth = depth
        self.q_device = tq.QuantumDevice(n_wires=n_wires)
        # Parameters: depth × n_wires
        self.params = nn.Parameter(torch.randn(depth, n_wires))

    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        # Encode x
        for d in range(self.depth):
            for w in range(self.n_wires):
                tq.ry(q_device, wires=w, params=x[:, w] * self.params[d, w])
        # Encode y with negative angles
        for d in reversed(range(self.depth)):
            for w in range(self.n_wires):
                tq.ry(q_device, wires=w, params=-y[:, w] * self.params[d, w])

class QuantumKernel(tq.QuantumModule):
    """Quantum kernel that evaluates the overlap of two encoded states.

    The kernel value is ``|⟨ψ(x)|ψ(y)⟩|`` where the states are prepared by
    :class:`VariationalAnsatz`.  The returned tensor has shape ``(batch,)``.
    """
    def __init__(self, n_wires: int = 4, depth: int = 2) -> None:
        super().__init__()
        self.ansatz = VariationalAnsatz(n_wires, depth)
        self.q_device = tq.QuantumDevice(n_wires=n_wires)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        # Return absolute amplitude of |0...0⟩
        return torch.abs(self.q_device.states.view(-1)[0])

# --- 4. Kernel selector ------------------------------------------------------
class KernelSelector(nn.Module):
    """Choose between a classical or quantum kernel at runtime.

    Parameters
    ----------
    method : str
        ``'classical'`` or ``'quantum'``.
    kwargs : dict
        Additional arguments forwarded to the chosen kernel constructor.
    """
    def __init__(self, method: str = "classical", **kwargs) -> None:
        super().__init__()
        if method == "classical":
            self.kernel = RBFKernel(**kwargs)
        elif method == "quantum":
            self.kernel = QuantumKernel(**kwargs)
        else:
            raise ValueError(f"Unsupported method {method!r}")

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.kernel(x, y)

# --- 5. Post‑processing network ---------------------------------------------
class PostProcessor(nn.Module):
    """Simple feed‑forward network that maps a kernel matrix to a scalar.

    Useful for regression tasks where the kernel matrix is treated as a feature
    vector (flattened or aggregated).
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, k_matrix: torch.Tensor) -> torch.Tensor:
        # Flatten the matrix into a vector per sample
        flat = k_matrix.flatten(start_dim=1)
        return self.net(flat)

# --- 6. Convenience kernel matrix -------------------------------------------
def kernel_matrix(a: Sequence[torch.Tensor],
                  b: Sequence[torch.Tensor],
                  gamma: float = 1.0,
                  method: str = "classical") -> np.ndarray:
    """
    Compute the Gram matrix between two collections of tensors.

    Parameters
    ----------
    a, b : Sequence[torch.Tensor]
        Collections of 1‑D tensors.  Each tensor is treated as a single sample.
    gamma : float, optional
        RBF decay parameter (used only for the classical kernel).
    method : str
        ``'classical'`` or ``'quantum'``.

    Returns
    -------
    np.ndarray
        The Gram matrix of shape ``(len(a), len(b))``.
    """
    if method == "classical":
        kernel = RBFKernel(gamma)
    elif method == "quantum":
        kernel = QuantumKernel()
    else:
        raise ValueError(f"Unsupported method {method!r}")

    # Compute pairwise values
    return np.array([[kernel(x.unsqueeze(0), y.unsqueeze(0)).item() for y in b] for x in a])

__all__ = [
    "RBFKernel",
    "KernalAnsatz",
    "Kernel",
    "VariationalAnsatz",
    "QuantumKernel",
    "KernelSelector",
    "PostProcessor",
    "kernel_matrix",
]
