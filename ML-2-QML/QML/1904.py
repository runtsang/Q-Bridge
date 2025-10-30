import pennylane as qml
import torch
import numpy as np


class QuantumHybridLayer(torch.nn.Module):
    """Variational quantum layer implemented with Pennylane, producing a single expectation value."""
    def __init__(self, n_qubits: int = 3, shift: float = 3.1415926535 / 2, shots: int = 1024):
        super().__init__()
        self.n_qubits = n_qubits
        self.shift = shift
        self.shots = shots
        self.dev = qml.device("default.qubit", wires=n_qubits, shots=shots)

        @qml.qnode(self.dev, interface="torch")
        def circuit(params: torch.Tensor) -> torch.Tensor:
            # Apply a global Hadamard layer
            for w in range(n_qubits):
                qml.Hadamard(wires=w)
            qml.barrier(wires=range(n_qubits))
            # Parameterised Ry rotations
            for idx, p in enumerate(params):
                qml.RY(p, wires=idx)
            # Expectation of Pauli-Z on the first qubit
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        params : torch.Tensor
            Tensor of shape [batch, 1] containing the classical logits to be fed into the quantum circuit.
            Each value will be replicated across all qubits.

        Returns
        -------
        torch.Tensor
            Expectation values of shape [batch, 1] produced by the variational circuit.
        """
        if params.ndim == 1:
            params = params.unsqueeze(0)  # shape: [1, 1]
        # Replicate the scalar logit across all qubits
        params_expanded = params.repeat(1, self.n_qubits)  # shape: [batch, n_qubits]
        # Use list comprehension to evaluate each quantum instance
        outputs = torch.stack([self.circuit(p) for p in params_expanded])
        return outputs


__all__ = ["QuantumHybridLayer"]
