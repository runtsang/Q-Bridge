"""Quantum kernel construction using a trainable variational ansatz."""

from __future__ import annotations

from typing import Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from torchquantum.functional import func_name_dict


class VariationalAnsatz(tq.QuantumModule):
    """
    Parameterised ansatz that encodes classical features into a quantum state
    and applies a trainable layer of two‑qubit entangling gates.

    Parameters
    ----------
    n_wires : int
        Number of qubits used for the encoding.
    depth : int, default 1
        Number of repetitions of the rotation‑entangle block.
    """

    def __init__(self, n_wires: int, depth: int = 1) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.depth = depth
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)

        # Create a list of rotation angles; each rotation is a trainable
        # parameter. The total number of parameters is n_wires * depth.
        self.params = nn.Parameter(torch.randn(self.n_wires * self.depth))

    @tq.static_support
    def forward(self, x: torch.Tensor) -> None:
        """
        Encode the input vector ``x`` into the quantum device and apply the
        variational circuit.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, n_wires) – classical data to encode.
        """
        batch_size = x.shape[0]
        self.q_device.reset_states(batch_size)

        # Encode each feature as an RY rotation.
        for i in range(self.n_wires):
            tq.ry(self.q_device, wires=[i], params=x[:, i])

        # Apply a trainable rotation‑entangle block.
        idx = 0
        for _ in range(self.depth):
            for i in range(self.n_wires - 1):
                tq.rx(self.q_device, wires=[i], params=self.params[idx])
                idx += 1
                tq.cnot(self.q_device, wires=[i, i + 1])
            # Wrap around to entangle the last and first qubit
            tq.rx(self.q_device, wires=[self.n_wires - 1], params=self.params[idx])
            idx += 1
            tq.cnot(self.q_device, wires=[self.n_wires - 1, 0])


class QuantumKernel(tq.QuantumModule):
    """
    Quantum kernel that evaluates the absolute overlap between two encoded
    quantum states produced by a :class:`VariationalAnsatz`.

    Parameters
    ----------
    n_wires : int, default 4
        Number of qubits used for the encoding.
    depth : int, default 1
        Depth of the variational circuit.
    """

    def __init__(self, n_wires: int = 4, depth: int = 1) -> None:
        super().__init__()
        self.ansatz = VariationalAnsatz(n_wires=n_wires, depth=depth)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the quantum kernel value for two batches of inputs.

        Parameters
        ----------
        x, y : torch.Tensor
            Shape (N, n_wires) – batches of classical data.

        Returns
        -------
        torch.Tensor
            Shape (N,) – kernel values for each pair (x_i, y_i).
        """
        # Encode x and y separately and compute the overlap.
        self.ansatz(x)
        states_x = self.ansatz.q_device.states.clone()

        self.ansatz(y)
        states_y = self.ansatz.q_device.states.clone()

        # Overlap amplitude: |⟨ψ_x|ψ_y⟩|
        overlap = torch.abs(torch.sum(states_x.conj() * states_y, dim=-1))
        return overlap

    def kernel_matrix(
        self,
        a: Union[Sequence[torch.Tensor], torch.Tensor],
        b: Union[Sequence[torch.Tensor], torch.Tensor],
    ) -> np.ndarray:
        """
        Compute the Gram matrix between two sets of samples.

        Parameters
        ----------
        a, b : sequence or tensor
            Input data. If a single tensor is supplied, it is treated as a
            batch of samples.

        Returns
        -------
        np.ndarray
            Kernel matrix of shape (len(a), len(b)).
        """
        if isinstance(a, torch.Tensor):
            a_tensor = a
        else:
            a_tensor = torch.stack(a, dim=0)
        if isinstance(b, torch.Tensor):
            b_tensor = b
        else:
            b_tensor = torch.stack(b, dim=0)

        # Compute pairwise kernel values.
        n_a, n_b = a_tensor.shape[0], b_tensor.shape[0]
        gram = torch.empty((n_a, n_b), device=a_tensor.device)

        for i in range(n_a):
            for j in range(n_b):
                gram[i, j] = self.forward(a_tensor[i : i + 1], b_tensor[j : j + 1])

        return gram.detach().cpu().numpy()


def kernel_matrix(
    a: Union[Sequence[torch.Tensor], torch.Tensor],
    b: Union[Sequence[torch.Tensor], torch.Tensor],
    n_wires: int = 4,
    depth: int = 1,
) -> np.ndarray:
    """
    Convenience wrapper that constructs a :class:`QuantumKernel` and returns
    the Gram matrix as a NumPy array.

    Parameters
    ----------
    a, b : sequence or tensor
        Input data.
    n_wires : int, default 4
        Number of qubits used for the encoding.
    depth : int, default 1
        Depth of the variational circuit.

    Returns
    -------
    np.ndarray
        Kernel matrix of shape (len(a), len(b)).
    """
    kernel = QuantumKernel(n_wires=n_wires, depth=depth)
    return kernel.kernel_matrix(a, b)


__all__ = ["VariationalAnsatz", "QuantumKernel", "kernel_matrix"]
