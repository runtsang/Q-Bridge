"""Quantum‑enhanced Conv‑Transformer model.

The module re‑implements the hybrid architecture with quantum sub‑modules.
It uses a quantum convolutional filter (QuanvCircuit) to extract patch
embeddings, and a transformer encoder where the feed‑forward network
is a quantum neural network.  The attention remains classical to keep
the design lightweight.  The public interface is identical to the
classical version so that the same training loop can be used.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from qiskit.circuit.random import random_circuit
from qiskit import Aer, execute
from typing import Tuple

# --------------------------------------------------------------------------- #
# 1. Quantum convolutional filter (quanvolution).
# --------------------------------------------------------------------------- #
class QuanvCircuit(nn.Module):
    """Quantum filter that maps a 2‑D kernel to a scalar via a quantum circuit."""
    def __init__(self, kernel_size: int, threshold: float = 127.0, shots: int = 100):
        super().__init__()
        self.n_qubits = kernel_size ** 2
        self.threshold = threshold
        self.shots = shots
        self.backend = Aer.get_backend("qasm_simulator")
        self.circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self.circuit.rx(self.theta[i], i)
        self.circuit.barrier()
        self.circuit += random_circuit(self.n_qubits, 2)
        self.circuit.measure_all()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Patch of shape (B, kernel_size, kernel_size).

        Returns
        -------
        torch.Tensor
            Tensor of shape (B,) containing the average |1> probability.
        """
        B = x.shape[0]
        data = x.reshape(B, -1).cpu().numpy()
        param_binds = []
        for dat in data:
            bind = {theta: np.pi if val > self.threshold else 0
                    for theta, val in zip(self.theta, dat)}
            param_binds.append(bind)
        job = execute(self.circuit, self.backend, shots=self.shots,
                      parameter_binds=param_binds)
        result = job.result().get_counts(self.circuit)
        probs = []
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            probs.append(ones * val / self.shots)
        return torch.tensor(probs, device=x.device, dtype=torch.float32)

# --------------------------------------------------------------------------- #
# 2. Quantum feed‑forward network (QNN).
# --------------------------------------------------------------------------- #
class FeedForwardQuantum(nn.Module):
    """Two‑layer quantum feed‑forward block using qiskit circuits."""
    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int, shots: int = 100):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.n_qubits = n_qubits
        self.shots = shots
        # Prepare a simple ansatz
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        self.params = [qiskit.circuit.Parameter(f"p{i}") for i in range(n_qubits)]
        # Encode input
        for i in range(n_qubits):
            self.circuit.rx(self.params[i], i)
        # Entangling layer
        for i in range(n_qubits - 1):
            self.circuit.cx(i, i + 1)
        self.circuit.cx(n_qubits - 1, 0)
        self.circuit.measure_all()
        self.backend = Aer.get_backend("qasm_simulator")
        # Classical linear layers
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape (B, embed_dim).

        Returns
        -------
        torch.Tensor
            Tensor of shape (B, embed_dim).
        """
        B = x.shape[0]
        probs = []
        for i in range(B):
            token = x[i].cpu().numpy()[:self.n_qubits]
            bind = {p: float(val) for p, val in zip(self.params, token)}
            job = execute(self.circuit, self.backend, shots=self.shots,
                          parameter_binds=[bind])
            result = job.result().get_counts(self.circuit)
            prob = 0.0
            for key, val in result.items():
                ones = sum(int(bit) for bit in key)
                prob += ones * val / self.shots
            probs.append(prob)
        q_out = torch.tensor(probs, device=x.device,
                             dtype=torch.float32).unsqueeze(-1)
        out = self.linear1(q_out)
        out = F.relu(out)
        out = self.linear2(out)
        return out

# --------------------------------------------------------------------------- #
# 3. Classical attention (kept simple).
# --------------------------------------------------------------------------- #
class MultiHeadAttentionClassical(nn.Module):
    """Standard multi‑head attention implemented classically."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                          dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, key_padding_mask=mask)
        return attn_out

# --------------------------------------------------------------------------- #
# 4. Transformer block with quantum feed‑forward.
# --------------------------------------------------------------------------- #
class TransformerBlockQuantum(nn.Module):
    """Transformer block where the feed‑forward network is quantum."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 n_qubits_ffn: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits_ffn)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

# --------------------------------------------------------------------------- #
# 5. Positional encoder (same as classical).
# --------------------------------------------------------------------------- #
class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) *
                             (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

# --------------------------------------------------------------------------- #
# 6. Quantum‑enhanced hybrid model.
# --------------------------------------------------------------------------- #
class ConvTransformerHybrid(nn.Module):
    """Hybrid model that uses a quantum convolutional filter followed by
    a transformer encoder with quantum feed‑forward layers."""
    def __init__(
        self,
        img_shape: Tuple[int, int],
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        kernel_size: int = 2,
        threshold: float = 127.0,
        n_qubits_ffn: int = 4,
        dropout: float = 0.1,
        shots: int = 100,
    ):
        super().__init__()
        self.img_shape = img_shape
        self.embed_dim = embed_dim
        self.conv = QuanvCircuit(kernel_size, threshold, shots)
        self.proj = nn.Linear(1, embed_dim)
        self.pos_enc = PositionalEncoder(embed_dim)
        self.transformer = nn.Sequential(
            *[TransformerBlockQuantum(embed_dim, num_heads, ffn_dim,
                                      n_qubits_ffn, dropout)
              for _ in range(num_blocks)]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Batch of images of shape (B, H, W).

        Returns
        -------
        torch.Tensor
            Logits of shape (B, num_classes).
        """
        # Quantum convolutional tokenization
        conv_out = self.conv(x)                 # (B, 1, H-k+1, W-k+1)
        B = conv_out.size(0)
        tokens = conv_out.view(B, 1, -1).transpose(1, 2)  # (B, L, 1)
        tokens = self.proj(tokens)               # (B, L, embed_dim)
        tokens = self.pos_enc(tokens)
        tokens = self.transformer(tokens)        # (B, L, embed_dim)
        pooled = tokens.mean(dim=1)              # global average pooling
        out = self.dropout(pooled)
        return self.classifier(out)

__all__ = ["ConvTransformerHybrid"]
