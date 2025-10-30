import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
import numpy as np


class DropoutScheduler:
    """Linear decay scheduler for dropout probability."""
    def __init__(self, init_dropout: float, min_dropout: float = 0.0, decay_steps: int = 1000):
        self.init_dropout = init_dropout
        self.min_dropout = min_dropout
        self.decay_steps = decay_steps
        self.step = 0

    def step_update(self) -> float:
        self.step += 1
        ratio = min(self.step / self.decay_steps, 1.0)
        return self.init_dropout - ratio * (self.init_dropout - self.min_dropout)

    def reset(self):
        self.step = 0


class FeedForward(nn.Module):
    """Standard two‑layer feed‑forward network for classical blocks."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

    def l1_regularization(self) -> torch.Tensor:
        return torch.tensor(0.0)


class QuantumAttention(nn.Module):
    """
    Multi‑head attention where each head performs a variational quantum circuit
    on a slice of the embedding. The circuit is parameterised by a trainable
    weight matrix that is regularised via an L1 penalty.
    """
    def __init__(self, embed_dim: int, num_heads: int, n_qubits: int = 4, n_layers: int = 2):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # Parameters for the variational circuit (one per qubit per layer)
        self.params = nn.Parameter(torch.randn(n_layers, n_qubits, 3))
        self.device = qml.device("default.qubit", wires=n_qubits, shots=512)

        # Classical linear maps
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=False)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def _quantum_block(self, x: torch.Tensor) -> torch.Tensor:
        @qml.qnode(self.device, interface="torch")
        def circuit(inputs, params):
            qml.templates.AngleEmbedding(inputs, wires=range(self.n_qubits))
            for l in range(self.n_layers):
                for q in range(self.n_qubits):
                    qml.Rot(params[l, q, 0], params[l, q, 1], params[l, q, 2], wires=q)
                qml.CNOT(wires=[0, 1])  # simple entangling pattern
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        return circuit(x, self.params)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq_len, _ = x.size()
        k = self.k_linear(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        q = self.q_linear(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Quantum perturbation on each head
        for h in range(self.num_heads):
            head_k = k[:, h, :, :].contiguous().view(batch * seq_len, self.d_k)
            head_q = q[:, h, :, :].contiguous().view(batch * seq_len, self.d_k)
            head_v = v[:, h, :, :].contiguous().view(batch * seq_len, self.d_k)

            # Reduce dimensionality to match qubits
            proj_k = head_k[:, : self.n_qubits]
            proj_q = head_q[:, : self.n_qubits]
            proj_v = head_v[:, : self.n_qubits]

            k_q = self._quantum_block(proj_k).view(batch, seq_len, self.n_qubits)
            q_q = self._quantum_block(proj_q).view(batch, seq_len, self.n_qubits)
            v_q = self._quantum_block(proj_v).view(batch, seq_len, self.n_qubits)

            # Upsample back to d_k
            k[:, h, :, :self.n_qubits] = k_q
            q[:, h, :, :self.n_qubits] = q_q
            v[:, h, :, :self.n_qubits] = v_q

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)
        return self.out_proj(attn_output)

    def l1_regularization(self) -> torch.Tensor:
        return torch.sum(torch.abs(self.params))


class QuantumFeedForward(nn.Module):
    """Two‑layer feed‑forward with a quantum projection on the hidden state."""
    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int = 4, n_layers: int = 2):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.params = nn.Parameter(torch.randn(n_layers, n_qubits, 3))
        self.device = qml.device("default.qubit", wires=n_qubits, shots=512)

        self.linear1 = nn.Linear(embed_dim, ffn_dim, bias=False)
        self.linear2 = nn.Linear(ffn_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(0.1)

    def _quantum_block(self, x: torch.Tensor) -> torch.Tensor:
        @qml.qnode(self.device, interface="torch")
        def circuit(inputs, params):
            qml.templates.AngleEmbedding(inputs, wires=range(self.n_qubits))
            for l in range(self.n_layers):
                for q in range(self.n_qubits):
                    qml.Rot(params[l, q, 0], params[l, q, 1], params[l, q, 2], wires=q)
                qml.CNOT(wires=[0, 1])
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        return circuit(x, self.params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = []
        for token in x.unbind(dim=1):
            token_q = token[:, :self.n_qubits]
            out = self._quantum_block(token_q)
            proj.append(out)
        proj = torch.stack(proj, dim=1)
        proj = self.linear1(self.dropout(proj))
        return self.linear2(F.relu(proj))

    def l1_regularization(self) -> torch.Tensor:
        return torch.sum(torch.abs(self.params))


class TransformerBlock(nn.Module):
    """Transformer block with optional quantum sub‑modules."""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        use_quantum: bool = False,
        quantum_reg_weight: float = 0.0,
        n_qubits: int = 4,
        n_layers: int = 2,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.use_quantum = use_quantum
        if use_quantum:
            self.attn = QuantumAttention(embed_dim, num_heads, n_qubits, n_layers)
            self.ffn = QuantumFeedForward(embed_dim, ffn_dim, n_qubits, n_layers)
        else:
            self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
            self.ffn = FeedForward(embed_dim, ffn_dim)
        self.quantum_reg_weight = quantum_reg_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_quantum:
            attn_out = self.attn(x)
        else:
            attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        return self.norm2(x + ffn_out)

    def l1_regularization(self) -> torch.Tensor:
        if self.use_quantum:
            return self.attn.l1_regularization() + self.ffn.l1_regularization()
        return torch.tensor(0.0)


class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TextClassifier(nn.Module):
    """
    Quantum‑enhanced transformer classifier.
    When `use_quantum=True` the model employs variational quantum circuits for the attention
    and feed‑forward components. Quantum parameters are regularised with an L1 penalty
    controlled by `quantum_reg_weight`.
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        use_quantum: bool = False,
        dropout_decay_steps: int = 1000,
        quantum_reg_weight: float = 0.0,
        n_qubits: int = 4,
        n_layers: int = 2,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim,
                    num_heads,
                    ffn_dim,
                    use_quantum=use_quantum,
                    quantum_reg_weight=quantum_reg_weight,
                    n_qubits=n_qubits,
                    n_layers=n_layers,
                )
                for _ in range(num_blocks)
            ]
        )
        self.dropout_scheduler = DropoutScheduler(dropout, min_dropout=0.0, decay_steps=dropout_decay_steps)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)
        self.use_quantum = use_quantum

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_encoder(tokens)
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=1)
        current_dropout = self.dropout_scheduler.step_update()
        x = nn.Dropout(current_dropout)(x)
        return self.classifier(x)

    def reset_dropout(self):
        """Reset the dropout scheduler to the initial state."""
        self.dropout_scheduler.reset()

    def quantum_regularization(self) -> torch.Tensor:
        """Sum of L1 penalties over all quantum blocks."""
        reg = sum(block.l1_regularization() for block in self.blocks if block.use_quantum)
        return reg * self.blocks[0].quantum_reg_weight if reg.numel() > 0 else torch.tensor(0.0)


__all__ = [
    "DropoutScheduler",
    "FeedForward",
    "QuantumAttention",
    "QuantumFeedForward",
    "TransformerBlock",
    "PositionalEncoder",
    "TextClassifier",
]
