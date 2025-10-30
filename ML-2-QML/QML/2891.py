"""Quantum kernel module for HybridNATModel."""
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict


class QuantumKernel(tq.QuantumModule):
    """Parameterized RY feature map with entanglement, returning overlap kernel."""

    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        # Build a simple entangling circuit: Ry on each wire followed by a chain of CX gates.
        self.ansatz = [
            tq.RY(has_params=True, trainable=False) for _ in range(n_wires)
        ] + [
            tq.CX() for _ in range(n_wires - 1)
        ]

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the overlap kernel between two input vectors x and y.
        x, y: 1‑D tensors of length n_wires.
        """
        # Create a fresh device for each evaluation to keep state clean.
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=1, device=x.device)
        # Encode x with positive parameters.
        for idx, gate in enumerate(self.ansatz[: self.n_wires]):
            gate(qdev, params=x[0, idx])
        # Uncompute y with negative parameters (reverse order).
        for idx, gate in enumerate(reversed(self.ansatz[: self.n_wires])):
            gate(qdev, params=-y[0, idx])
        # Return the absolute value of the overlap of the two states.
        return torch.abs(qdev.states.view(-1)[0])

    def kernel_matrix(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Compute the Gram matrix between two sets of vectors.
        a: shape (n, n_wires), b: shape (m, n_wires)
        """
        n = a.shape[0]
        m = b.shape[0]
        K = torch.zeros((n, m), device=a.device)
        for i in range(n):
            for j in range(m):
                K[i, j] = self.forward(a[i], b[j])
        return K


__all__ = ["QuantumKernel"]
