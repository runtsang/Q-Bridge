"""Quantum self‑attention circuit using Pennylane.

The circuit implements a variational self‑attention block.  Rotation
parameters map to single‑qubit rotations, while entangle parameters
control the depth of a fixed CZ‑entanglement pattern.  The output is
an expectation value that can be used as a differentiable attention
score.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
import torch


class SelfAttention:
    """
    Variational self‑attention block built on Pennylane.
    Parameters
    ----------
    n_qubits : int
        Number of qubits (must be at least 2).
    """

    def __init__(self, n_qubits: int) -> None:
        self.n_qubits = n_qubits
        if n_qubits < 2:
            raise ValueError("n_qubits must be >= 2")
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch")

    def _circuit(
        self,
        rotation_params: torch.Tensor,
        entangle_params: torch.Tensor,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        rotation_params : torch.Tensor
            Shape (3*n_qubits,) – rotation angles for RX,RY,RZ on each qubit.
        entangle_params : torch.Tensor
            Shape (n_qubits-1,) – phase shifts applied after CZ gates.
        inputs : torch.Tensor
            Shape (n_qubits,) – binary encoding of the input sequence.
        """
        # Encode input bits as phase shifts
        for i in range(self.n_qubits):
            qml.PhaseShift(inputs[i] * np.pi, wires=i)

        # Single‑qubit rotations (parameterised)
        for i in range(self.n_qubits):
            qml.RX(rotation_params[3 * i], wires=i)
            qml.RY(rotation_params[3 * i + 1], wires=i)
            qml.RZ(rotation_params[3 * i + 2], wires=i)

        # Fixed entanglement pattern (CZ gates)
        for i in range(self.n_qubits - 1):
            qml.CZ(wires=[i, i + 1])
            # Optional tunable phase after entanglement
            qml.PhaseShift(entangle_params[i], wires=i + 1)

        # Output observable – expectation of PauliZ on the last qubit
        return qml.expval(qml.PauliZ(self.n_qubits - 1))

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> float:
        """
        Evaluate the variational attention circuit.

        Parameters
        ----------
        rotation_params : np.ndarray
            Rotation angles, shape (3*n_qubits,).
        entangle_params : np.ndarray
            Entanglement phase shifts, shape (n_qubits-1,).
        inputs : np.ndarray
            Binary input array, shape (n_qubits,).

        Returns
        -------
        float
            Expectation value of the chosen observable.
        """
        rot = torch.tensor(rotation_params, dtype=torch.float32, requires_grad=False)
        ent = torch.tensor(entangle_params, dtype=torch.float32, requires_grad=False)
        inp = torch.tensor(inputs, dtype=torch.float32, requires_grad=False)
        return self.qnode(rot, ent, inp).item()

    def gradient(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute gradients of the expectation value w.r.t. the parameters
        using Pennylane's autograd.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Gradients for rotation_params and entangle_params.
        """
        rot = torch.tensor(rotation_params, dtype=torch.float32, requires_grad=True)
        ent = torch.tensor(entangle_params, dtype=torch.float32, requires_grad=True)
        inp = torch.tensor(inputs, dtype=torch.float32, requires_grad=False)
        val = self.qnode(rot, ent, inp)
        val.backward()
        return rot.grad.numpy(), ent.grad.numpy()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_qubits={self.n_qubits})"


__all__ = ["SelfAttention"]
