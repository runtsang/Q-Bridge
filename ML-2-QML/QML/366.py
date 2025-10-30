"""Quantum kernel construction with a learnable variational ansatz."""

from __future__ import annotations

from typing import Sequence, Optional

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq

class QuantumKernelMethod(tq.QuantumModule):
    """
    Variational quantum kernel that learns the encoding circuit.
    The circuit consists of alternating layers of parameterised RY rotations
    and CNOT entanglers. The parameters are optimised to maximise kernel
    alignment with a target Gram matrix.
    """

    def __init__(
        self,
        n_wires: int = 4,
        n_layers: int = 2,
        init_params: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.n_wires = n_wires
        self.n_layers = n_layers
        self.params = nn.Parameter(
            init_params
            if init_params is not None
            else torch.randn(n_layers, n_wires, requires_grad=True)
        )
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)

    def _encode(
        self, x: torch.Tensor, sign: float = 1.0, reset: bool = True
    ) -> None:
        """
        Encode a batch of classical data into the quantum state.
        The encoding uses a simple RY rotation per qubit followed by a
        layer of CNOTs to entangle the qubits. The rotation angles are learned.
        """
        if reset:
            self.q_device.reset_states(x.shape[0])
        for layer in range(self.n_layers):
            for w in range(self.n_wires):
                params = sign * x[:, w] * self.params[layer, w]
                tq.RY(self.q_device, wires=[w], params=params)
            for w in range(self.n_wires):
                tq.CNOT(
                    self.q_device, wires=[w, (w + 1) % self.n_wires]
                )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the overlap <ψ(x)|ψ(y)> by encoding x, then y with
        inverted parameters, and measuring the probability of
        the |0...0> state.
        """
        x = x.reshape(-1, self.n_wires)
        y = y.reshape(-1, self.n_wires)

        self._encode(x, sign=1.0, reset=True)
        self._encode(y, sign=-1.0, reset=False)

        return torch.abs(self.q_device.states.view(-1)[0])

    def kernel_matrix(
        self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]
    ) -> np.ndarray:
        """
        Compute the Gram matrix between two lists of tensors in a batched way.
        """
        a = torch.stack(a)
        b = torch.stack(b)
        return self.forward(a, b).detach().cpu().numpy()

    def train(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        target_kernel: torch.Tensor,
        lr: float = 1e-2,
        epochs: int = 200,
        verbose: bool = False,
    ) -> None:
        """
        Train the variational parameters to maximise kernel alignment with
        a target Gram matrix. The loss is the negative alignment:
        L = - trace(K_q K_t) / sqrt(trace(K_q^2) trace(K_t^2))
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        for epoch in range(epochs):
            optimizer.zero_grad()
            K_q = self.forward(x, y)
            alignment = torch.trace(K_q @ target_kernel) / torch.sqrt(
                torch.trace(K_q @ K_q) * torch.trace(target_kernel @ target_kernel)
            )
            loss = -alignment
            loss.backward()
            optimizer.step()
            if verbose and (epoch % (epochs // 10) == 0 or epoch == epochs - 1):
                print(
                    f"Epoch {epoch+1}/{epochs} loss={loss.item():.6f} alignment={alignment.item():.6f}"
                )

__all__ = ["QuantumKernelMethod"]
