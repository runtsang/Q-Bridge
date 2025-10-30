"""Quantum self‑attention using a Pennylane variational circuit."""

import pennylane as qml
import numpy as np
import torch


class SelfAttentionModule:
    """
    Variational quantum circuit that produces an attention matrix from
    expectation values of Pauli‑Z on each qubit.

    Parameters
    ----------
    n_qubits : int, default 4
        Number of qubits / attention heads.
    wires : list[int] | None, default None
        Wires to use on the device; defaults to 0…n_qubits‑1.
    """

    def __init__(self, n_qubits: int = 4, wires: list[int] | None = None):
        self.n_qubits = n_qubits
        self.wires = wires or list(range(n_qubits))
        self.dev = qml.device("default.qubit", wires=self.wires)

        # Shapes of learnable parameters
        self.rotation_params_shape = (n_qubits, 3)   # RX, RY, RZ per qubit
        self.entangle_params_shape = (n_qubits - 1,)  # CRX between adjacent qubits

        @qml.qnode(self.dev, interface="torch")
        def circuit(rot_params: torch.Tensor, ent_params: torch.Tensor):
            # Rotation layer
            for i in range(self.n_qubits):
                qml.RX(rot_params[i, 0], wires=self.wires[i])
                qml.RY(rot_params[i, 1], wires=self.wires[i])
                qml.RZ(rot_params[i, 2], wires=self.wires[i])

            # Entanglement layer
            for i in range(self.n_qubits - 1):
                qml.CRX(ent_params[i], wires=[self.wires[i], self.wires[i + 1]])

            # Measure expectation values of Pauli‑Z on each qubit
            return [qml.expval(qml.PauliZ(w)) for w in self.wires]

        self.circuit = circuit

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ) -> np.ndarray:
        """
        Execute the circuit and return a normalized attention matrix.

        Parameters
        ----------
        rotation_params : np.ndarray
            Shape (n_qubits, 3) – rotation angles for each qubit.
        entangle_params : np.ndarray
            Shape (n_qubits-1,) – entanglement angles for adjacent qubits.
        shots : int, default 1024
            Number of shots for the simulator.

        Returns
        -------
        np.ndarray
            Attention matrix of shape (n_qubits, n_qubits) where each row
            sums to one.
        """
        # Convert inputs to torch tensors
        rot = torch.tensor(rotation_params, dtype=torch.float32)
        ent = torch.tensor(entangle_params, dtype=torch.float32)

        # Set number of shots
        self.dev.shots = shots

        # Execute circuit
        expectation = self.circuit(rot, ent).detach().numpy()

        # Map expectation values from [-1,1] to [0,1]
        scores = (expectation + 1) / 2.0

        # Build attention matrix via outer product and row‑normalise
        attn = np.outer(scores, scores)
        attn = attn / attn.sum(axis=-1, keepdims=True)
        return attn
