"""Quantum SelfAttention module implemented with PennyLane.

Key features:
- Variational circuit with rotation and entanglement layers per qubit.
- Parameter‑shift gradients for differentiable training.
- Supports batched inputs and returns Z‑expectation values for each qubit.
"""

import pennylane as qml
import numpy as np
import torch


class SelfAttention:
    """
    Quantum self‑attention block built with PennyLane.
    """

    def __init__(self, n_qubits: int = 4, wires: list | None = None):
        self.n_qubits = n_qubits
        self.wires = wires if wires is not None else list(range(n_qubits))
        # Device with a fixed shot count for reproducibility
        self.dev = qml.device("default.qubit", wires=self.wires, shots=1024)

        # Initialize parameters for rotation and entanglement layers
        self.rotation_params = np.random.randn(n_qubits, 3)  # RX, RY, RZ per qubit
        self.entangle_params = np.random.randn(n_qubits - 1)  # CRX between neighbors

    def _quantum_circuit(self,
                         rotation_params: torch.Tensor,
                         entangle_params: torch.Tensor,
                         inputs: torch.Tensor) -> list:
        """Builds the quantum circuit for a single input vector."""
        for i, wire in enumerate(self.wires):
            # Encode input as a rotation around Z
            qml.RZ(inputs[i], wires=wire)
            # Apply parameterized rotations
            qml.RX(rotation_params[i, 0], wires=wire)
            qml.RY(rotation_params[i, 1], wires=wire)
            qml.RZ(rotation_params[i, 2], wires=wire)

        # Entanglement layer
        for i in range(self.n_qubits - 1):
            qml.CRX(entangle_params[i], wires=[self.wires[i], self.wires[i + 1]])

        # Measure expectation of Z on each qubit
        return [qml.expval(qml.PauliZ(w)) for w in self.wires]

    @qml.qnode(device=lambda: self.dev, interface="torch")
    def _qnode(self,
               rotation_params: torch.Tensor,
               entangle_params: torch.Tensor,
               inputs: torch.Tensor) -> list:
        return self._quantum_circuit(rotation_params, entangle_params, inputs)

    def run(self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            inputs: np.ndarray,
            batch_size: int = 1) -> torch.Tensor:
        """
        Execute the variational circuit on a batch of inputs.

        Parameters
        ----------
        rotation_params : np.ndarray shape (n_qubits, 3)
        entangle_params : np.ndarray shape (n_qubits-1,)
        inputs : np.ndarray shape (batch_size, n_qubits)
        batch_size : int

        Returns
        -------
        torch.Tensor of shape (batch_size, n_qubits) containing expectation values.
        """
        outputs = []
        for inp in inputs:
            out = self._qnode(torch.tensor(rotation_params, dtype=torch.float32),
                              torch.tensor(entangle_params, dtype=torch.float32),
                              torch.tensor(inp, dtype=torch.float32))
            outputs.append(out)
        return torch.stack(outputs)

    def get_params(self) -> dict:
        """Return current parameters as a dictionary."""
        return {"rotation_params": self.rotation_params,
                "entangle_params": self.entangle_params}
