"""
Quantum UnifiedAttentionTransformer using Pennylane.
Author: gpt-oss-20b
"""

import pennylane as pn
import pennylane.numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Simple positional encoder
class PositionalEncoder(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class UnifiedAttentionTransformer(nn.Module):
    """
    Quantum‑enhanced transformer that uses Pennylane for the attention block.
    """
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int = 2,
                 device: str = "cpu"):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.ffn_dim = ffn_dim
        self.num_classes = num_classes
        self.device = device

        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)

        # Classical feed‑forward layers
        self.ffn1 = nn.Linear(embed_dim, ffn_dim)
        self.ffn2 = nn.Linear(ffn_dim, embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

        # PennyLane device and circuit
        self.qdev = pn.device("default.qubit", wires=embed_dim)

        def attention_circuit(inputs, rotation_params, entangle_params):
            # inputs: (embed_dim,)
            for i in range(self.embed_dim):
                pn.rx(rotation_params[i], wires=i)
                pn.ry(rotation_params[self.embed_dim + i], wires=i)
                pn.rz(rotation_params[2 * self.embed_dim + i], wires=i)
            for i in range(self.embed_dim - 1):
                pn.cnot(wires=[i, i+1])
            pn.measure_all()
            return pn.probs(wires=range(self.embed_dim))

        self.attention_qnode = pn.QNode(attention_circuit, self.qdev, interface="torch")

    def forward(self, input_ids, rotation_params=None, entangle_params=None):
        x = self.token_embedding(input_ids)
        x = self.pos_encoder(x)
        for _ in range(self.num_blocks):
            # Quantum attention applied token‑wise
            quantum_out = []
            for token in x.unbind(dim=1):
                # flatten token to 1‑D
                token_flat = token.view(-1)
                # If parameters are not provided, use identity
                if rotation_params is None:
                    rotation_params = torch.zeros(3 * self.embed_dim, device=self.device)
                if entangle_params is None:
                    entangle_params = torch.zeros(self.embed_dim - 1, device=self.device)
                q_out = self.attention_qnode(token_flat, rotation_params, entangle_params)
                quantum_out.append(q_out.unsqueeze(1))
            x = torch.cat(quantum_out, dim=1)

            # Classical feed‑forward after quantum attention
            x = F.relu(self.ffn1(x))
            x = self.ffn2(x)

        x = x.mean(dim=1)
        x = self.classifier(x)
        return x

__all__ = ["UnifiedAttentionTransformer"]
