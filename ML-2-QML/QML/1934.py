"""SelfAttentionModule: quantum‑augmented attention using a Pennylane variational circuit."""
import pennylane as qml
import torch
import torch.nn as nn
import numpy as np


class SelfAttentionModule(nn.Module):
    """
    Hybrid self‑attention block that replaces classical attention weights
    with a quantum‑derived probability vector.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input embeddings.
    num_heads : int, default=1
        Number of attention heads.
    num_qubits : int, default=4
        Number of qubits used in the variational circuit.
    device : str or torch.device, default="cpu"
        Target device for torch tensors.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 1,
        num_qubits: int = 4,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_qubits = num_qubits
        self.device = device

        # Classical linear projections (no bias)
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3, bias=False)

        # Quantum parameters
        self.rotation_params = nn.Parameter(
            torch.randn(num_qubits * 3, device=self.device)
        )
        self.entangle_params = nn.Parameter(
            torch.randn(num_qubits - 1, device=self.device)
        )

        # Quantum device (simulator)
        self.qcdev = qml.device("default.qubit", wires=num_qubits, shots=1024)

    def quantum_weights(self) -> torch.Tensor:
        """
        Runs the variational circuit and returns a probability vector.
        The vector is normalised to sum to one and is used to modulate
        attention weights.

        Returns
        -------
        torch.Tensor
            Probability vector of shape (2**num_qubits,).
        """

        @qml.qnode(self.qcdev, interface="torch")
        def circuit():
            for i in range(self.num_qubits):
                qml.RX(self.rotation_params[3 * i], wires=i)
                qml.RY(self.rotation_params[3 * i + 1], wires=i)
                qml.RZ(self.rotation_params[3 * i + 2], wires=i)
            for i in range(self.num_qubits - 1):
                qml.CRX(self.entangle_params[i], wires=[i, i + 1])
            return qml.probs()

        probs = circuit()
        # Normalise to avoid numerical drift
        return probs / probs.sum()

    def forward(
        self,
        x: torch.Tensor,
        attn_params: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq, embed_dim).
        attn_params : torch.Tensor
            Element‑wise weighting for the QKV projections, shape
            (batch, seq, 3 * embed_dim).
        mask : torch.Tensor, optional
            Boolean mask broadcastable to (batch, seq, seq).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, seq, embed_dim).
        """
        batch, seq, _ = x.size()

        # Classical QKV projection with custom params
        qkv = self.qkv_proj(x)
        qkv = qkv * attn_params
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        # Reshape for multi‑head
        q = q.view(batch, seq, self.num_heads, self.embed_dim // self.num_heads).transpose(1, 2)
        k = k.view(batch, seq, self.num_heads, self.embed_dim // self.num_heads).transpose(1, 2)
        v = v.view(batch, seq, self.num_heads, self.embed_dim // self.num_heads).transpose(1, 2)

        # Scaled dot‑product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.embed_dim // self.num_heads, dtype=torch.float32)
        )
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1), float("-inf"))
        attn_weights = torch.softmax(scores, dim=-1)

        # Quantum‑derived modulation
        q_weights = self.quantum_weights().to(x.device)  # (2**num_qubits,)
        # Use only the first `seq` entries; pad if necessary
        if seq > len(q_weights):
            pad = torch.zeros(seq - len(q_weights), device=x.device)
            q_weights = torch.cat([q_weights, pad])
        else:
            q_weights = q_weights[:seq]
        q_weights = q_weights / q_weights.sum()
        # Broadcast across batch, heads, seq
        q_weights = q_weights.unsqueeze(0).unsqueeze(0).unsqueeze(2)
        attn_weights = attn_weights * q_weights
        attn_weights = attn_weights / attn_weights.sum(-1, keepdim=True)

        weighted = torch.matmul(attn_weights, v)
        weighted = weighted.transpose(1, 2).contiguous().view(batch, seq, self.embed_dim)
        # No residual or norm; output is directly the weighted sum
        return weighted

__all__ = ["SelfAttentionModule"]
