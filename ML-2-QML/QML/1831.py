"""Quantum self‑attention implemented with Pennylane.

The circuit consists of parameterised single‑qubit rotations followed
by a fixed entangling layer.  The output is a vector of expectation
values that can be interpreted as the attention weights.  A simple
parameter‑shift gradient routine is provided for training.
"""

import pennylane as qml
import numpy as np
import torch
from typing import Tuple, Callable


class SelfAttentionEnhanced:
    """
    Quantum self‑attention with a variational circuit.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (typically equal to the sequence length).
    """

    def __init__(self, n_qubits: int) -> None:
        self.n_qubits = n_qubits
        self.dev = qml.device("default.qubit", wires=n_qubits)

        # Parameter containers
        # Rotation angles for each qubit: (n_qubits, 3)
        self.rotation_params = np.random.uniform(0, 2 * np.pi, size=(n_qubits, 3))
        # Entangling parameters for consecutive qubits: (n_qubits-1,)
        self.entangle_params = np.random.uniform(0, 2 * np.pi, size=(n_qubits - 1,))

        # Define QNode
        @qml.qnode(self.dev, interface="torch")
        def circuit(rot, ent):
            for i in range(self.n_qubits):
                qml.RX(rot[i, 0], wires=i)
                qml.RY(rot[i, 1], wires=i)
                qml.RZ(rot[i, 2], wires=i)
            for i in range(self.n_qubits - 1):
                qml.CRX(ent[i], wires=[i, i + 1])
            return [qml.expval(qml.Z(i)) for i in range(self.n_qubits)]

        self.circuit = circuit

    def forward(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> np.ndarray:
        """
        Run the variational circuit and return expectation values.

        Parameters
        ----------
        rotation_params : np.ndarray
            Shape (n_qubits, 3) rotation angles.
        entangle_params : np.ndarray
            Shape (n_qubits-1,) entangling angles.

        Returns
        -------
        np.ndarray
            Expectation values of Z on each qubit.
        """
        # Convert to torch tensors for Pennylane
        rot_t = torch.tensor(rotation_params, dtype=torch.float64, requires_grad=False)
        ent_t = torch.tensor(entangle_params, dtype=torch.float64, requires_grad=False)
        out = self.circuit(rot_t, ent_t)
        return out.detach().numpy()

    def parameter_shift_gradient(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        loss_fn: Callable[[np.ndarray], float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute gradients of the loss w.r.t. circuit parameters using the
        parameter‑shift rule.

        Parameters
        ----------
        rotation_params : np.ndarray
            Current rotation angles.
        entangle_params : np.ndarray
            Current entangling angles.
        loss_fn : callable
            Function that maps circuit output to a scalar loss.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Gradients for rotation and entangling parameters.
        """
        eps = np.pi / 2
        grad_rot = np.zeros_like(rotation_params)
        grad_ent = np.zeros_like(entangle_params)

        # Rotation parameters
        for i in range(self.n_qubits):
            for j in range(3):
                shift = np.zeros_like(rotation_params)
                shift[i, j] = eps
                out_plus = self.forward(rotation_params + shift, entangle_params)
                out_minus = self.forward(rotation_params - shift, entangle_params)
                grad_rot[i, j] = (loss_fn(out_plus) - loss_fn(out_minus)) / (2 * np.sin(eps))

        # Entangling parameters
        for i in range(self.n_qubits - 1):
            shift = np.zeros_like(entangle_params)
            shift[i] = eps
            out_plus = self.forward(rotation_params, entangle_params + shift)
            out_minus = self.forward(rotation_params, entangle_params - shift)
            grad_ent[i] = (loss_fn(out_plus) - loss_fn(out_minus)) / (2 * np.sin(eps))

        return grad_rot, grad_ent

    def train_step(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        loss_fn: Callable[[np.ndarray], float],
        lr: float = 0.01,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Perform one gradient‑descent step.

        Parameters
        ----------
        rotation_params : np.ndarray
            Current rotation angles.
        entangle_params : np.ndarray
            Current entangling angles.
        loss_fn : callable
            Loss function mapping circuit output to a scalar.
        lr : float
            Learning rate.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, float]
            Updated parameters and loss value.
        """
        grad_rot, grad_ent = self.parameter_shift_gradient(rotation_params, entangle_params, loss_fn)
        new_rot = rotation_params - lr * grad_rot
        new_ent = entangle_params - lr * grad_ent
        loss_val = loss_fn(self.forward(new_rot, new_ent))
        return new_rot, new_ent, loss_val


__all__ = ["SelfAttentionEnhanced"]
