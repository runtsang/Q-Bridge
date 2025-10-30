"""Classical multi‑head self‑attention with optional hybrid quantum mode.

The design extends the seed by:
* Multi‑head support.
* Learnable linear projections for Q, K, V.
* Optional hybrid mode that delegates attention weight generation to a Pennylane variational circuit.
"""

import numpy as np
import torch
import pennylane as qml
from typing import Optional

class SelfAttention:
    """
    Multi‑head self‑attention block.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of input embeddings.
    num_heads : int, default 4
        Number of attention heads.
    device : str, default 'cpu'
        Torch device.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, device: str = "cpu"):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.head_dim = embed_dim // num_heads
        self.device = device

        # Trainable linear projections
        self.W_q = torch.nn.Parameter(torch.randn(embed_dim, embed_dim, device=device))
        self.W_k = torch.nn.Parameter(torch.randn(embed_dim, embed_dim, device=device))
        self.W_v = torch.nn.Parameter(torch.randn(embed_dim, embed_dim, device=device))

        # Placeholder for hybrid quantum circuit
        self._quantum_circuit = None

    def _init_quantum_circuit(self, num_qubits: int):
        """Create a simple Pennylane variational circuit for attention weights."""
        dev = qml.device("default.qubit", wires=num_qubits)

        @qml.qnode(dev, interface="torch")
        def circuit(params: torch.Tensor):
            for i in range(num_qubits):
                qml.RX(params[3 * i], wires=i)
                qml.RY(params[3 * i + 1], wires=i)
                qml.RZ(params[3 * i + 2], wires=i)
            for i in range(num_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            # Return expectation of PauliZ on first qubit as a scalar
            return qml.expval(qml.PauliZ(0))

        self._quantum_circuit = circuit

    def run(
        self,
        inputs: torch.Tensor,
        mode: str = "classic",
        quantum_params: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute self‑attention.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape (batch, seq_len, embed_dim).
        mode : {"classic", "hybrid"}, default "classic"
            Whether to use classical attention or hybrid quantum‑derived weights.
        quantum_params : torch.Tensor, optional
            Parameters for the variational circuit (required if mode="hybrid").

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, seq_len, embed_dim).
        """
        batch, seq_len, _ = inputs.shape

        # Linear projections
        Q = torch.matmul(inputs, self.W_q)  # (batch, seq_len, embed_dim)
        K = torch.matmul(inputs, self.W_k)
        V = torch.matmul(inputs, self.W_v)

        # Reshape for multi‑head
        Q = Q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot‑product attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)

        if mode == "classic":
            attn_weights = torch.softmax(scores, dim=-1)
        elif mode == "hybrid":
            if quantum_params is None:
                raise ValueError("quantum_params must be provided in hybrid mode")
            if self._quantum_circuit is None:
                self._init_quantum_circuit(num_qubits=seq_len)
            # Generate a scalar weight from the circuit
            q_scalar = self._quantum_circuit(quantum_params)
            # Broadcast to match batch, heads, seq_len, seq_len
            attn_weights = q_scalar.expand(batch, self.num_heads, seq_len, seq_len)
        else:
            raise ValueError(f"Unsupported mode {mode}")

        context = torch.matmul(attn_weights, V)  # (batch, num_heads, seq_len, head_dim)
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)
        return context
