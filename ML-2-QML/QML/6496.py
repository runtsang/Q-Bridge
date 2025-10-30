"""Quantum Self‑Attention using Pennylane with variational circuits."""
from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

class SelfAttention:
    """
    Quantum self‑attention block based on a variational circuit.

    Parameters
    ----------
    embed_dim : int
        Number of qubits (also the dimensionality of the token space).
    device_name : str, optional
        Pennylane device backend. Defaults to ``'default.qubit'``.
    """

    def __init__(self, embed_dim: int, device_name: str = "default.qubit") -> None:
        self.embed_dim = embed_dim
        self.device = qml.device(device_name, wires=embed_dim)
        # Pre‑allocate parameter arrays to avoid re‑allocation in each run
        self.rotation_params = np.zeros((embed_dim, 3))
        self.entangle_params = np.zeros(embed_dim - 1)

    def _circuit(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        input_state: np.ndarray,
    ):
        """Variational circuit that encodes the input and applies attention‑like
        rotations and entanglement."""
        # Encode the classical input into qubit states
        for i in range(self.embed_dim):
            qml.RY(pnp.arccos(2 * input_state[i] - 1), wires=i)

        # Parameterized single‑qubit rotations
        for i in range(self.embed_dim):
            qml.RX(rotation_params[i, 0], wires=i)
            qml.RY(rotation_params[i, 1], wires=i)
            qml.RZ(rotation_params[i, 2], wires=i)

        # Entangling layer (controlled‑RX gates)
        for i in range(self.embed_dim - 1):
            qml.CRX(entangle_params[i], wires=[i, i + 1])

        # Measurement in computational basis
        return qml.probs(wires=range(self.embed_dim))

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
        shots: int = 1024,
    ) -> np.ndarray:
        """
        Execute the variational circuit for each input token and return
        probability distributions over the 2^embed_dim possible outcomes.

        Parameters
        ----------
        rotation_params : np.ndarray
            Shape (embed_dim, 3) – rotation angles for each qubit.
        entangle_params : np.ndarray
            Shape (embed_dim-1,) – angles for the controlled‑RX gates.
        inputs : np.ndarray
            Array of shape (batch, embed_dim) with values in [0, 1] that
            encode the token.
        shots : int, optional
            Number of measurement shots. Defaults to 1024.

        Returns
        -------
        np.ndarray
            Probabilities of shape (batch, 2**embed_dim).
        """
        probs_list = []
        for token in inputs:
            # Wrap the circuit in a QNode
            qnode = qml.QNode(
                lambda rp, ep, inp: self._circuit(rp, ep, inp),
                self.device,
                interface="autograd",
            )
            # Execute with specified shots
            probs = qnode(rotation_params, entangle_params, token)
            probs_list.append(probs.numpy())

        return np.stack(probs_list)

__all__ = ["SelfAttention"]
