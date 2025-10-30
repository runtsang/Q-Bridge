"""Quantum kernel implementation using TorchQuantum with a variational ansatz."""

from __future__ import annotations

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
import torch.nn as nn
from typing import Sequence, Optional

class QuantumKernelMethod__gen125(tq.QuantumModule):
    """Quantum RBF‑style kernel built from a trainable variational circuit."""
    
    def __init__(
        self,
        n_wires: int = 4,
        depth: int = 2,
        basis_gates: Sequence[str] | None = None,
        device: str | None = None,
    ):
        """
        Parameters
        ----------
        n_wires : int
            Number of qubits.
        depth : int
            Number of repeating blocks in the ansatz.
        basis_gates : sequence of str, optional
            Rotations used in each block.  Defaults to ``['ry', 'rz']``.
        device : str or torch.device, optional
            Device on which the quantum device is allocated. Defaults to ``'cpu'``.
        """
        super().__init__()
        self.n_wires = n_wires
        self.depth = depth
        self.basis_gates = basis_gates or ["ry", "rz"]
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires, device=device)
        self.ansatz = self._build_ansatz()
    
    def _build_ansatz(self) -> tq.QuantumModule:
        """
        Construct a depth‑wise variational circuit.
        Each block consists of a rotation on every wire followed by a linear chain of CNOTs.
        """
        class VariationalAnsatz(tq.QuantumModule):
            def __init__(self, outer: "QuantumKernelMethod__gen125"):
                super().__init__()
                self.outer = outer
                # One trainable angle per rotation gate per block
                self.params = nn.Parameter(
                    torch.randn(outer.depth * outer.n_wires, 1)
                )
            
            @tq.static_support
            def forward(
                self,
                q_device: tq.QuantumDevice,
                x: torch.Tensor,
                y: torch.Tensor,
            ) -> None:
                # Encode x with a rotation on each wire
                for d in range(self.outer.depth):
                    for w in range(self.outer.n_wires):
                        gate = self.outer.basis_gates[d % len(self.outer.basis_gates)]
                        idx = d * self.outer.n_wires + w
                        # data‑dependent rotation
                        data_rot = x[:, w]
                        # trainable offset
                        train_rot = self.params[idx]
                        func_name_dict[gate](
                            q_device,
                            wires=[w],
                            params=data_rot + train_rot,
                        )
                    # entangling layer
                    for w in range(self.outer.n_wires - 1):
                        func_name_dict["cx"](q_device, wires=[w, w + 1])
                
                # Reverse encoding of y with negative sign
                for d in reversed(range(self.outer.depth)):
                    for w in range(self.outer.n_wires):
                        gate = self.outer.basis_gates[d % len(self.outer.basis_gates)]
                        idx = d * self.outer.n_wires + w
                        data_rot = -y[:, w]
                        train_rot = self.params[idx]
                        func_name_dict[gate](
                            q_device,
                            wires=[w],
                            params=data_rot + train_rot,
                        )
                    for w in range(self.outer.n_wires - 1):
                        func_name_dict["cx"](q_device, wires=[w, w + 1])
        
        return VariationalAnsatz(self)
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the kernel for two 1‑D tensors.
        
        Parameters
        ----------
        x, y : torch.Tensor
            1‑D tensors of the same shape.
            
        Returns
        -------
        torch.Tensor
            Kernel value (scalar).
        """
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        # Overlap of the final state with the all‑zero computational basis
        return torch.abs(self.q_device.states.view(-1)[0])
    
    def kernel_matrix(
        self,
        a: Sequence[torch.Tensor],
        b: Sequence[torch.Tensor],
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute the Gram matrix between two collections of tensors.
        
        Parameters
        ----------
        a, b : Sequence[torch.Tensor]
            Sequences of 1‑D tensors.
        batch_size : int, optional
            If provided, compute the matrix in mini‑batches to reduce peak memory.
            
        Returns
        -------
        numpy.ndarray
            2‑D array of shape (len(a), len(b)).
        """
        a = [x.reshape(1, -1) for x in a]
        b = [y.reshape(1, -1) for y in b]
        n_a, n_b = len(a), len(b)
        if batch_size is None or batch_size >= n_a:
            K = torch.empty((n_a, n_b), device=self.q_device.device)
            for i, xi in enumerate(a):
                for j, yj in enumerate(b):
                    K[i, j] = self.forward(xi.squeeze(), yj.squeeze())
            return K.cpu().numpy()
        else:
            K = torch.empty((n_a, n_b), device=self.q_device.device)
            for i in range(0, n_a, batch_size):
                end = min(i + batch_size, n_a)
                for ii in range(i, end):
                    xi = a[ii]
                    for j, yj in enumerate(b):
                        K[ii, j] = self.forward(xi.squeeze(), yj.squeeze())
            return K.cpu().numpy()
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(n_wires={self.n_wires}, depth={self.depth}, "
            f"basis_gates={self.basis_gates}, device={self.q_device.device})"
        )

__all__ = ["QuantumKernelMethod__gen125"]
