"""Quantum self‑attention using Pennylane with a graph entangler.

The class `SelfAttention__gen263` mirrors the classical API but operates on a
parameter‑driven variational circuit.  It accepts rotation and entanglement
parameters, encodes the input features into rotations, applies a customizable
entangling layer, and returns differentiable expectation values that can be
used as attention scores.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
import torch
from pennylane import numpy as pnp

class SelfAttention__gen263:
    """
    Quantum self‑attention.

    Parameters
    ----------
    embed_dim : int
        Number of qubits (must equal the input dimensionality).
    device_name : str, default "default.qubit"
        Pennylane device to use.
    """

    def __init__(self, embed_dim: int, device_name: str = "default.qubit"):
        self.embed_dim = embed_dim
        self.dev = qml.device(device_name, wires=embed_dim, shots=1024)

        # Define the QNode
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(self, rot_params, ent_params, inputs):
            # Feature map: encode classical inputs into rotations
            for i in range(self.embed_dim):
                qml.RX(inputs[i], wires=i)
            # Parameterized single‑qubit rotations
            for i in range(self.embed_dim):
                qml.RX(rot_params[0, i], wires=i)
                qml.RY(rot_params[1, i], wires=i)
                qml.RZ(rot_params[2, i], wires=i)
            # Graph entangler
            for i in range(self.embed_dim):
                for j in range(i + 1, self.embed_dim):
                    if ent_params[i, j] > 0:
                        qml.CRY(ent_params[i, j], wires=[i, j])
            # Measure expectation of PauliZ for each qubit
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.embed_dim)]

        self.circuit = circuit

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the quantum attention output.

        Parameters
        ----------
        rotation_params : np.ndarray
            Shape ``(3, embed_dim)`` – parameters for RX, RY, RZ per qubit.
        entangle_params : np.ndarray
            Symmetric adjacency matrix of shape ``(embed_dim, embed_dim)``.
            Entry (i, j) controls the angle of a CRY gate between qubits i and j.
        inputs : np.ndarray
            Classical input vector of shape ``(embed_dim,)``.

        Returns
        -------
        np.ndarray
            Expectation values of PauliZ on each qubit, shape ``(embed_dim,)``.
        """
        # Convert to torch tensors for autograd
        rot = torch.as_tensor(rotation_params, dtype=torch.float32, requires_grad=True)
        ent = torch.as_tensor(entangle_params, dtype=torch.float32)
        inp = torch.as_tensor(inputs, dtype=torch.float32)

        # Run the circuit
        out = self.circuit(rot, ent, inp)
        # Convert to NumPy and detach
        return out.detach().cpu().numpy()

__all__ = ["SelfAttention__gen263"]
