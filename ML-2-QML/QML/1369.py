"""Quantum self‑attention using a variational circuit.

The circuit implements a parameter‑shaped attention block that
produces a probability distribution over the input tokens.
"""

import pennylane as qml
import numpy as np
import torch


class SelfAttentionBlock:
    """Variational self‑attention block.

    Parameters
    ----------
    n_qubits : int
        Number of qubits, one per token in the sequence.
    dev : pennylane.Device, optional
        Pennylane device to use. Defaults to the qiskit Aer simulator.
    """

    def __init__(self, n_qubits: int, dev: qml.Device | None = None):
        self.n_qubits = n_qubits
        self.dev = dev or qml.device("qiskit.aer", wires=n_qubits, shots=1024)

        # Parameter shapes
        self.rotation_shape = (n_qubits, 3)   # RX, RY, RZ per qubit
        self.entangle_shape = (n_qubits - 1,)  # CX between adjacent qubits

        # Initialize parameters
        self.rotation_params = np.random.uniform(0, 2 * np.pi, self.rotation_shape)
        self.entangle_params = np.random.uniform(0, 2 * np.pi, self.entangle_shape)

        # Define a parameter‑shift differentiable circuit
        @qml.qnode(self.dev, interface="torch")
        def circuit(rotation_params, entangle_params):
            # Rotations
            for i in range(self.n_qubits):
                qml.RX(rotation_params[i, 0], wires=i)
                qml.RY(rotation_params[i, 1], wires=i)
                qml.RZ(rotation_params[i, 2], wires=i)
            # Entangling CX gates
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])

            # Measure expectation of Z on each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.circuit = circuit

    def run(
        self,
        inputs: torch.Tensor,
        rotation_params: np.ndarray | None = None,
        entangle_params: np.ndarray | None = None,
        shots: int | None = None,
    ) -> torch.Tensor:
        """
        Execute the variational attention circuit and return attention weights.

        Parameters
        ----------
        inputs : torch.Tensor
            Shape (batch, seq_len). Dummy placeholder to keep API compatibility.
        rotation_params : np.ndarray, optional
            Override default rotation parameters.
        entangle_params : np.ndarray, optional
            Override default entanglement parameters.
        shots : int, optional
            Number of shots for the simulator. Ignored when using a differentiable
            device.

        Returns
        -------
        torch.Tensor
            Attention weights of shape (batch, seq_len), normalised to sum to one.
        """
        # Use provided parameters or defaults
        rot = rotation_params if rotation_params is not None else self.rotation_params
        ent = entangle_params if entangle_params is not None else self.entangle_params

        # Execute circuit
        if shots is not None:
            self.dev.shots = shots
        probs = self.circuit(rot, ent)  # shape (n_qubits,)

        # Convert expectation values to probabilities via softmax
        probs = torch.tensor(probs, dtype=torch.float32)
        probs = torch.softmax(probs, dim=0)

        # Expand to batch dimension for compatibility
        return probs.unsqueeze(0).repeat(inputs.shape[0], 1)

__all__ = ["SelfAttentionBlock"]
