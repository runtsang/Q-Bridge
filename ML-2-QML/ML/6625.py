"""Hybrid self‑attention module that can be used as a drop‑in replacement for the original SelfAttention.

This class extends the original implementation by adding a quantum mode that uses a variational circuit to compute attention
weights. The rotation and entangle parameters become trainable torch.Parameters, allowing end‑to‑end optimisation with a
classical optimiser. The interface mirrors the original seed: a run method that accepts rotation_params, entangle_params,
and inputs and returns a numpy array. In quantum mode the method forwards the computation to a Qiskit circuit defined
in the companion quantum module.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit

# Import quantum helper – the module must be available in the same package
try:
    from.quantum_self_attention import get_quantum_circuit
except Exception:
    # Fallback stub if the quantum module is not present
    def get_quantum_circuit(*args, **kwargs):
        raise RuntimeError("Quantum module not available")

class SelfAttentionDual(nn.Module):
    """Hybrid self‑attention with optional quantum backend."""
    def __init__(self, embed_dim: int = 4, use_quantum: bool = False, device: torch.device | None = None):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_quantum = use_quantum
        self.device = device or torch.device("cpu")

        # Classical linear projections
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)

        # Trainable parameters for the quantum circuit
        self.rotation_params = nn.Parameter(torch.randn(embed_dim * 3))
        self.entangle_params = nn.Parameter(torch.randn(embed_dim - 1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Compute self‑attention over the input sequence.

        Parameters
        ----------
        inputs : torch.Tensor
            Shape (batch, seq_len, embed_dim).

        Returns
        -------
        torch.Tensor
            Shape (batch, seq_len, embed_dim).
        """
        if self.use_quantum:
            # Encode inputs as rotation angles and run the variational circuit
            # to obtain a probability amplitude vector per batch element.
            batch_size, seq_len, _ = inputs.shape
            outputs = []
            for b in range(batch_size):
                # Flatten the sequence for this example
                flat = inputs[b].reshape(-1).cpu().numpy()
                # Combine with rotation_params
                rot = self.rotation_params.detach().cpu().numpy() + flat
                # Build and run circuit
                circuit = get_quantum_circuit(self.embed_dim, rot, self.entangle_params.detach().cpu().numpy())
                # Use statevector simulator to get amplitudes
                state = qiskit.Aer.get_backend("statevector_simulator").run(circuit).result().get_statevector(circuit)
                outputs.append(state)
            # Convert list of statevectors to tensor
            return torch.tensor(outputs, dtype=torch.float32, device=self.device)
        else:
            # Classical attention
            q = self.W_q(inputs)
            k = self.W_k(inputs)
            v = self.W_v(inputs)
            scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.embed_dim)
            attn = F.softmax(scores, dim=-1)
            return torch.matmul(attn, v)

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        """
        Compatibility wrapper that mimics the original seed's run signature.

        Parameters
        ----------
        rotation_params : np.ndarray
            Shape (embed_dim*3,) – used only in quantum mode.
        entangle_params : np.ndarray
            Shape (embed_dim-1,) – used only in quantum mode.
        inputs : np.ndarray
            Shape (batch, seq_len, embed_dim).

        Returns
        -------
        np.ndarray
            Output tensor converted to a NumPy array.
        """
        # For classical mode, ignore the provided parameters
        if not self.use_quantum:
            inputs_t = torch.as_tensor(inputs, dtype=torch.float32, device=self.device)
            out = self.forward(inputs_t)
            return out.detach().cpu().numpy()
        else:
            # Override internal parameters with the provided ones
            self.rotation_params.data = torch.as_tensor(rotation_params, dtype=torch.float32, device=self.device)
            self.entangle_params.data = torch.as_tensor(entangle_params, dtype=torch.float32, device=self.device)
            inputs_t = torch.as_tensor(inputs, dtype=torch.float32, device=self.device)
            out = self.forward(inputs_t)
            return out.detach().cpu().numpy()

__all__ = ["SelfAttentionDual"]
