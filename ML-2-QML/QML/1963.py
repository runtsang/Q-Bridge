"""Quantum RBF kernel with a trainable variational ansatz and minibatch evaluation.

The implementation relies on torchquantum and can be trained to approximate a classical
RBF kernel.  The kernel is computed as the absolute overlap of the final quantum state
with the reference state.  The ansatz uses a parametric Ry rotation on each qubit
followed by a layer of CNOTs.  All rotation angles are trainable and are
multiplied by the input feature values.

The class exposes a :meth:`fit` method that optimises the parameters by
minimising the mean‑squared error between the quantum kernel matrix and a
classical RBF kernel with a user‑specified or auto‑tuned bandwidth.  The
kernel matrix is evaluated in minibatches to keep memory usage bounded.
"""

from __future__ import annotations

from typing import Sequence, Iterable, Optional

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
from torch import nn, optim

class KernalAnsatz(tq.QuantumModule):
    """
    Variational ansatz that encodes classical data via
    trainable rotations.  For each input feature a Ry rotation
    with a trainable weight is applied to the corresponding qubit.
    A fixed layer of CNOTs entangles the qubits.
    """
    def __init__(self, n_wires: int) -> None:
        super().__init__()
        self.n_wires = n_wires
        # Trainable weights for each qubit
        self.weights = nn.Parameter(torch.randn(n_wires))
        # Entanglement pattern
        self.cnot_pattern = [(i, (i + 1) % n_wires) for i in range(n_wires)]

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor) -> None:
        """
        Apply the variational circuit for the batch of data ``x``.
        """
        q_device.reset_states(x.shape[0])
        # Encode data
        for i in range(self.n_wires):
            params = x[:, i] * self.weights[i]
            func_name_dict["ry"](q_device, wires=[i], params=params)
        # Entangle
        for control, target in self.cnot_pattern:
            func_name_dict["cnot"](q_device, wires=[control, target])

class QuantumKernelMethod(tq.QuantumModule):
    """
    Quantum kernel module that evaluates the similarity between
    two feature vectors via a trainable variational ansatz.
    The kernel value is |<0|ψ(x)> <0|ψ(y)>|^2 which reduces to the
    absolute overlap of the final states.

    Parameters
    ----------
    n_wires : int, default 4
        Number of qubits used for encoding.
    gamma : float, default 1.0
        Classical RBF bandwidth used as the target during training.
    batch_size : int, default 200
        Batch size used when computing the Gram matrix.
    """
    def __init__(
        self,
        n_wires: int = 4,
        gamma: float = 1.0,
        batch_size: int = 200,
    ) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=n_wires)
        self.ansatz = KernalAnsatz(n_wires)
        self.gamma = gamma
        self.batch_size = batch_size

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        """
        Compute the quantum kernel for a pair of batches ``x`` and ``y``.
        The circuit is run twice: once for each input, and the final
        state overlaps are returned.
        """
        # Encode first batch
        self.ansatz(q_device, x)
        # Store the states
        states_x = q_device.states.clone()
        # Encode second batch
        self.ansatz(q_device, y)
        states_y = q_device.states.clone()
        # Compute absolute overlap
        overlap = torch.abs(torch.sum(states_x * states_y.conj(), dim=-1))
        q_device.states = overlap.unsqueeze(-1)

    def kernel_matrix(self, X: Sequence[np.ndarray], Y: Sequence[np.ndarray]) -> np.ndarray:
        """
        Compute the Gram matrix between X and Y in minibatches.
        """
        X_np = np.asarray(X)
        Y_np = np.asarray(Y)
        n, m = X_np.shape[0], Y_np.shape[0]
        K = np.empty((n, m), dtype=np.float64)

        for i in range(0, n, self.batch_size):
            X_batch = torch.tensor(X_np[i : i + self.batch_size], dtype=torch.float32)
            for j in range(0, m, self.batch_size):
                Y_batch = torch.tensor(Y_np[j : j + self.batch_size], dtype=torch.float32)
                self.forward(self.q_device, X_batch, Y_batch)
                K[i : i + X_batch.shape[0], j : j + Y_batch.shape[0]] = self.q_device.states.squeeze().numpy()
        return K

    def fit(
        self,
        X: Iterable[np.ndarray],
        y: Optional[Iterable[np.ndarray]] = None,
        lr: float = 0.01,
        epochs: int = 50,
    ) -> "QuantumKernelMethod":
        """
        Train the variational parameters to approximate a classical RBF
        kernel with bandwidth ``self.gamma``.  The loss is the mean‑squared
        error between the quantum kernel matrix and the classical one.

        Parameters
        ----------
        X : Iterable[np.ndarray]
            Training samples.
        y : Optional[Iterable[np.ndarray]]
            Ignored; present only for API compatibility.
        lr : float
            Learning rate for the Adam optimiser.
        epochs : int
            Number of optimisation epochs.

        Returns
        -------
        self
        """
        X_np = np.asarray(list(X))
        # Classical RBF kernel for supervision
        def classical_kernel(a, b):
            diff = a[:, None, :] - b[None, :, :]
            return np.exp(-self.gamma * np.sum(diff ** 2, axis=-1))

        target_K = classical_kernel(X_np, X_np)

        optimizer = optim.Adam(self.parameters(), lr=lr)
        loss_fct = nn.MSELoss()
        for _ in range(epochs):
            optimizer.zero_grad()
            K_pred = self.kernel_matrix(X_np, X_np)
            loss = loss_fct(torch.tensor(K_pred, dtype=torch.float32), torch.tensor(target_K, dtype=torch.float32))
            loss.backward()
            optimizer.step()
        return self

    def __call__(self, X: Sequence[np.ndarray], Y: Sequence[np.ndarray]) -> np.ndarray:
        return self.kernel_matrix(X, Y)

__all__ = ["QuantumKernelMethod", "KernalAnsatz"]
