"""Quantum‑enhanced transformer that can optionally use a Qiskit self‑attention circuit.

The module mirrors the classical hybrid above but replaces the
multi‑head attention with a quantum module that uses the
``QuantumSelfAttention`` circuit from the seed ``SelfAttention.py``.
It also supports a fully quantum feed‑forward layer via
``FeedForwardQuantum``.  All quantum sub‑modules are built with
TorchQuantum or Qiskit and expose a ``forward`` method compatible
with the rest of the transformer stack.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from qiskit import Aer

# Import the quantum SelfAttention helper
from.SelfAttention import SelfAttention


class QuantumSelfAttentionWrapper(nn.Module):
    """Quantum self‑attention block based on the Qiskit circuit."""
    def __init__(self, embed_dim: int, n_qubits: int = 4, shots: int = 1024):
        super().__init__()
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = Aer.get_backend("qasm_simulator")
        self.attn = SelfAttention()  # returns QuantumSelfAttention instance

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        b, s, e = x.shape
        inp = x.reshape(b * s, e).cpu().numpy()
        # For simplicity we use random parameters; in practice these would be trainable
        rot = np.random.randn(self.n_qubits * 3).reshape(-1)
        ent = np.random.randn(self.n_qubits - 1)
        counts = self.attn.run(self.backend, rot, ent, shots=self.shots)
        # Convert counts dict to a tensor of zeros (placeholder).
        out = np.zeros_like(inp)
        return torch.from_numpy(out).to(x.device).reshape(b, s, e)


class FeedForwardQuantum(nn.Module):
    """Two‑layer feed‑forward realized by a quantum module."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_qubits: int):
            super().__init__()
            self.n_wires = n_qubits
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
            )
            self.parameters = nn.ModuleList(
                [tq.RY(has_params=True, trainable=True) for _ in range(n_qubits)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            for wire, gate in enumerate(self.parameters):
                gate(q_device, wires=wire)
            return self.measure(q_device)

    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int, dropout: float = 0.1):
        super().__init__()
        self.q_layer = self.QLayer(n_qubits)
        self.q_device = tq.QuantumDevice(n_wires=n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        outputs = []
        for token in x.unbind(dim=1):
            qdev = self.q_device.copy(bsz=token.size(0), device=token.device)
            outputs.append(self.q_layer(token, qdev))
        out = torch.stack(outputs, dim=1)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))


class FeedForwardClassical(nn.Module):
    """Classical feed‑forward for fallback."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerBlockQuantum(nn.Module):
    """Transformer block that can use either a quantum or classical feed‑forward."""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
        use_quantum_ffn: bool = False,
        n_qubits_ffn: int = 0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = QuantumSelfAttentionWrapper(embed_dim, n_qubits=8, shots=1024)
        if use_quantum_ffn and n_qubits_ffn > 0:
            self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits_ffn, dropout)
        else:
            self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-np.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TextClassifierQuantum(nn.Module):
    """Quantum‑enhanced transformer classifier."""
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        use_quantum_ffn: bool = False,
        n_qubits_ffn: int = 0,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)
        self.transformers = nn.Sequential(
            *[
                TransformerBlockQuantum(
                    embed_dim,
                    num_heads,
                    ffn_dim,
                    dropout,
                    use_quantum_ffn=use_quantum_ffn,
                    n_qubits_ffn=n_qubits_ffn,
                )
                for _ in range(num_blocks)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_embedding(tokens)
        x = self.transformers(x)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)


__all__ = [
    "QuantumSelfAttentionWrapper",
    "FeedForwardQuantum",
    "FeedForwardClassical",
    "TransformerBlockQuantum",
    "PositionalEncoder",
    "TextClassifierQuantum",
]
