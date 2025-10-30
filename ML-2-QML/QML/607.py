"""Quantum RBF kernel with a trainable variational ansatz and gradient support."""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict


class VariationalRBFAnsatz(tq.QuantumModule):
    """
    Variational ansatz that encodes two classical data vectors via Ry rotations
    with trainable angles, followed by a fixed sequence of CNOT gates to create
    entanglement.  The overlap of the two resulting states yields the kernel value.

    Parameters
    ----------
    n_qubits : int, default=4
        Number of qubits used for encoding.
    """

    def __init__(self, n_qubits: int = 4) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        # Trainable parameters for the two data encodings
        self.theta_x = nn.Parameter(torch.randn(n_qubits))
        self.theta_y = nn.Parameter(torch.randn(n_qubits))
        # Fixed entangling circuit
        self.entangle = [
            {"func": "cx", "wires": [0, 1]},
            {"func": "cx", "wires": [1, 2]},
            {"func": "cx", "wires": [2, 3]},
        ]

    @tq.static_support
    def forward(
        self,
        q_device: tq.QuantumDevice,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> None:
        """
        Encode both data vectors and apply entanglement.

        Parameters
        ----------
        q_device : tq.QuantumDevice
            Quantum device on which to run the circuit.
        x, y : torch.Tensor
            1‑D tensors of length ``n_qubits`` containing classical data.
        """
        # Reset device and encode x
        q_device.reset_states(x.shape[0])
        for i in range(self.n_qubits):
            func_name_dict["ry"](q_device, wires=[i], params=x[:, i] + self.theta_x[i])

        # Apply entanglement
        for gate in self.entangle:
            func_name_dict[gate["func"]](
                q_device,
                wires=gate["wires"],
            )

        # Uncompute entanglement and encode y with opposite sign
        for gate in reversed(self.entangle):
            func_name_dict[gate["func"]](
                q_device,
                wires=gate["wires"],
            )
        for i in range(self.n_qubits):
            func_name_dict["ry"](q_device, wires=[i], params=-y[:, i] + self.theta_y[i])


class QuantumRBFKernel(tq.QuantumModule):
    """
    Quantum kernel that evaluates the overlap of two encoded states produced by
    :class:`VariationalRBFAnsatz`.  The module supports automatic differentiation
    via PyTorch autograd and can run on GPU if the underlying device supports it.

    Parameters
    ----------
    n_qubits : int, default=4
        Number of qubits for the kernel circuit.
    """

    def __init__(self, n_qubits: int = 4) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.q_device = tq.QuantumDevice(n_wires=n_qubits, device="cpu")
        self.ansatz = VariationalRBFAnsatz(n_qubits)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the kernel value between two batches of samples.

        Parameters
        ----------
        x, y : torch.Tensor
            1‑D tensors of shape (batch, n_qubits).

        Returns
        -------
        torch.Tensor
            Kernel values of shape (batch,).
        """
        x = x.reshape(-1, self.n_qubits)
        y = y.reshape(-1, self.n_qubits)
        self.ansatz(self.q_device, x, y)
        # The first element of the state vector represents the overlap amplitude
        return torch.abs(self.q_device.states.view(-1)[0])

    def kernel_matrix(
        self,
        a: Sequence[torch.Tensor],
        b: Sequence[torch.Tensor],
    ) -> np.ndarray:
        """
        Construct the Gram matrix between two lists of tensors.

        Parameters
        ----------
        a, b : Sequence[torch.Tensor]
            Sequences of 1‑D tensors of length ``n_qubits``.

        Returns
        -------
        np.ndarray
            Kernel matrix of shape (len(a), len(b)).
        """
        device = self.q_device.device
        # Batch the inputs for efficient evaluation
        a_stack = torch.stack(a).to(device)
        b_stack = torch.stack(b).to(device)
        n_a, n_b = a_stack.shape[0], b_stack.shape[0]
        matrix = torch.empty((n_a, n_b), device=device)
        for i in range(n_a):
            matrix[i] = self.forward(a_stack[i].unsqueeze(0), b_stack)
        return matrix.cpu().numpy()


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    """
    Convenience wrapper to compute the quantum kernel Gram matrix.

    Parameters
    ----------
    a, b : Sequence[torch.Tensor]
        Sequences of 1‑D tensors.

    Returns
    -------
    np.ndarray
        Gram matrix of shape (len(a), len(b)).
    """
    kernel = QuantumRBFKernel()
    return kernel.kernel_matrix(a, b)


__all__ = ["VariationalRBFAnsatz", "QuantumRBFKernel", "kernel_matrix"]
