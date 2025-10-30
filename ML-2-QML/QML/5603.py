"""Quantum kernel construction with a reusable, device‑aware ansatz."""

from __future__ import annotations

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict

__all__ = ["QuantumKernel", "kernel_matrix"]

# --------------------------------------------------------------------------- #
# Quantum kernel – a reusable, device‑aware module that can work with a
# fixed circuit or a variational ansatz.
# --------------------------------------------------------------------------- #
class QuantumKernel(tq.QuantumModule):
    """Quantum kernel using a TorchQuantum ansatz.

    Parameters
    ----------
    n_wires : int
        Number of qubits used to encode the data.
    depth : int, default 1
        The number of times the encoding and entangling gates
        are repeated.  This is the depth of the quantum circuit.
    QDevice : torchquantum.QuantumDevice | None
        The quantum device or simulator used to return the
        (x‑? s)‑based kernel value.
    """

    def __init__(self,
                 n_wires: int,
                 depth: int = 1,
                 QDevice: tq.QuantumDevice | None = None) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.depth = depth
        self.QDevice = QDevice
        self._build_ansatz()

    # ------------------------------------------------------------------ #
    # Build a reusable encoding circuit that can be tested with
    #   > python -m qiskit‑tune‑like ‑t <t> <t‑??>.
    # ------------------------------------------------------------------ #
    def _build_ansatz(self) -> None:
        """Create the data‑encoding and entangling circuit."""
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
        # Entangling pattern repeated `depth` times
        self.entangle = [
            {"func": "cnot", "wires": [i, (i + 1) % self.n_wires]}
            for i in range(self.n_wires)
        ]

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the absolute value of the overlap of the encoded states."""
        # Reshape for a single sample
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)

        # Initialise device if not already present
        if self.QDevice is None:
            self.QDevice = tq.QuantumDevice(n_wires=self.n_wires)

        # Encode first vector
        self.ansatz(self.QDevice, x)
        for _ in range(self.depth):
            for gate in self.entangle:
                func_name_dict[gate["func"]](self.QDevice, wires=gate["wires"])
        state1 = self.QDevice.states.clone()

        # Reset and encode second vector
        self.QDevice.reset_states(x.shape[0])
        self.ansatz(self.QDevice, y)
        for _ in range(self.depth):
            for gate in self.entangle:
                func_name_dict[gate["func"]](self.QDevice, wires=gate["wires"])

        # Overlap of two states
        overlap = self.QDevice.states @ state1.conj()
        return torch.abs(overlap).view(-1)

    def set_device(self, device: torch.device) -> None:
        """Allow the caller to specify a GPU device for the simulator."""
        self.QDevice = None  # reset to force re‑initialisation on next forward


# --------------------------------------------------------------------------- #
# Kernel matrix helper – vectorised for both kernels.
# --------------------------------------------------------------------------- #
def kernel_matrix(
    a: list[torch.Tensor],
    b: list[torch.Tensor],
    *,
    kernel: tq.QuantumModule | None = None,
    n_wires: int = 4,
    depth: int = 1,
) -> np.ndarray:
    """Compute the Gram matrix between two collections of tensors.

    Parameters
    ----------
    a, b : list[torch.Tensor]
        Input data in the shape ``(N, D)`` or ``(N, 1, D)``.
    kernel : tq.QuantumModule, optional
        Override the default quantum kernel.  If ``None`` a QuantumKernel
        instance is created.
    ``n_wires`` and ``depth`` are passed to the kernel if it
    is not already initialised.

    Returns
    -------
    np.ndarray
        The Gram matrix of shape ``(len(a), len(b))``.
    """
    if kernel is None:
        kernel = QuantumKernel(n_wires=n_wires, depth=depth)

    # Broadcast to 2‑D for easy broadcasting
    a = torch.stack(a).reshape(-1, a[0].shape[-1])
    b = torch.stack(b).reshape(-1, b[0].shape[-1])

    mat = torch.zeros(a.shape[0], b.shape[0], dtype=torch.float32)
    for i, xi in enumerate(a):
        for j, yj in enumerate(b):
            mat[i, j] = kernel(xi, yj).item()
    return mat.cpu().numpy()
