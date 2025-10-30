"""Quantum self‑attention module using Pennylane.

The implementation mirrors the classical API but replaces the
scaled‑dot‑product attention with a parameterised quantum circuit
whose measurement statistics are interpreted as attention scores.
"""

import pennylane as qml
import numpy as np
import torch
import torch.nn.functional as F

class SelfAttentionModule:
    def __init__(self, embed_dim: int, heads: int = 1, shots: int = 1024, device: str = "default.qubit"):
        self.embed_dim = embed_dim
        self.heads = heads
        self.shots = shots
        self.device = qml.device(device, wires=embed_dim)
        self.rotation_params = None
        self.entangle_params = None

    def _circuit(self, inputs, rotation_params, entangle_params):
        """Parameterized circuit that encodes the inputs and attention parameters."""
        for i in range(self.embed_dim):
            # Encode the classical input as an angle on the qubit
            qml.RX(np.arctan2(inputs[i], 1.0), wires=i)
            # Apply trainable rotations
            qml.RX(rotation_params[i, 0], wires=i)
            qml.RY(rotation_params[i, 1], wires=i)
            qml.RZ(rotation_params[i, 2], wires=i)

        # Entangle neighbouring qubits
        for i in range(self.embed_dim - 1):
            qml.CNOT(wires=[i, i + 1])
            qml.RX(entangle_params[i], wires=i)
            qml.CNOT(wires=[i, i + 1])

        # Measurement as expectation of PauliZ
        return [qml.expval(qml.PauliZ(i)) for i in range(self.embed_dim)]

    @qml.qnode(device, interface="torch")
    def _qnode(self, inputs, rotation_params, entangle_params):
        return self._circuit(inputs, rotation_params, entangle_params)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Compute the quantum attention output.

        Parameters
        ----------
        inputs : torch.Tensor
            Input of shape (batch, seq_len, embed_dim).

        Returns
        -------
        torch.Tensor
            Output of shape (batch, seq_len, embed_dim).
        """
        batch, seq_len, _ = inputs.shape
        outputs = []
        for b in range(batch):
            batch_out = []
            for s in range(seq_len):
                inp = inputs[b, s].detach().cpu().numpy()
                if self.rotation_params is None:
                    self.rotation_params = torch.randn(self.embed_dim, 3, device=inputs.device)
                if self.entangle_params is None:
                    self.entangle_params = torch.randn(self.embed_dim - 1, device=inputs.device)
                out = self._qnode(inp, self.rotation_params, self.entangle_params)
                batch_out.append(out)
            outputs.append(torch.stack(batch_out))
        return torch.stack(outputs)

    def run(self, inputs: torch.Tensor, rotation_params: np.ndarray, entangle_params: np.ndarray) -> torch.Tensor:
        """
        Public API matching the classical implementation.
        """
        rotation_params = torch.tensor(rotation_params, dtype=torch.float32, device=inputs.device)
        entangle_params = torch.tensor(entangle_params, dtype=torch.float32, device=inputs.device)
        return self.forward(inputs)

def SelfAttention(embed_dim: int, heads: int = 1, shots: int = 1024, device: str = "default.qubit"):
    """
    Factory returning a ready‑to‑use instance of SelfAttentionModule.
    """
    return SelfAttentionModule(embed_dim=embed_dim, heads=heads, shots=shots, device=device)

__all__ = ["SelfAttention", "SelfAttentionModule"]
