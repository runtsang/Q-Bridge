import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class MultiHeadAttentionQuantum(nn.Module):
    """Quantum implementation of multi‑head attention."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, q_device=None):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.q_device = q_device or tq.QuantumDevice(n_wires=self.head_dim, bsz=1)
        self.circuits = nn.ModuleList([self._build_circuit(self.head_dim) for _ in range(num_heads)])
        self.hooks = []

    def _build_circuit(self, n_wires: int):
        circuit = tq.QuantumModule()
        circuit.n_wires = n_wires
        circuit.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
        )
        circuit.parameters = nn.ModuleList(
            [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
        )
        circuit.measure = tq.MeasureAll(tq.PauliZ)
        return circuit

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        B, T, D = x.shape
        x = x.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # B,H,T,Dh
        outputs = []
        for h in range(self.num_heads):
            head_input = x[:, h]  # B,T,Dh
            flat = head_input.reshape(B * T, self.head_dim)
            qdev = self.q_device.copy(bsz=B * T, device=flat.device)
            qdev.reset()
            self.circuits[h].encoder(qdev, flat)
            for gate in self.circuits[h].parameters:
                gate(qdev, wires=list(range(self.head_dim)))
            out = self.circuits[h].measure(qdev)  # (B*T, Dh)
            out = out.reshape(B, T, self.head_dim)
            outputs.append(out)
        out = torch.stack(outputs, dim=1)  # B,H,T,Dh
        out = out.transpose(1, 2).contiguous().view(B, T, self.embed_dim)
        out = self.dropout(out)
        for hook in self.hooks:
            hook(out)
        return out

    def register_hook(self, hook_fn):
        self.hooks.append(hook_fn)


class FeedForwardQuantum(nn.Module):
    """Quantum feed‑forward network."""
    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int, dropout: float = 0.1):
        super().__init__()
        if n_qubits < 1:
            raise ValueError("n_qubits must be positive")
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.n_qubits = n_qubits
        self.dropout = nn.Dropout(dropout)
        self.q_device = tq.QuantumDevice(n_wires=n_qubits, bsz=1)
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
        )
        self.parameters = nn.ModuleList(
            [tq.RY(has_params=True, trainable=True) for _ in range(n_qubits)]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.hooks = []

    def forward(self, x: torch.Tensor):
        B, T, D = x.shape
        outputs = []
        for t in range(T):
            token = x[:, t]  # B,D
            qdev = self.q_device.copy(bsz=B, device=token.device)
            qdev.reset()
            self.encoder(qdev, token)
            for gate in self.parameters:
                gate(qdev, wires=list(range(self.n_qubits)))
            out = self.measure(qdev)  # (B, n_qubits)
            outputs.append(out)
        out = torch.stack(outputs, dim=1)  # B,T,n_qubits
        out = self.linear1(self.dropout(out))
        out = F.relu(out)
        out = self.linear2(out)
        for hook in self.hooks:
            hook(out)
        return out

    def register_hook(self, hook_fn):
        self.hooks.append(hook_fn)


class PositionalEncodingQuantum(nn.Module):
    """Quantum‑based positional encoding that outputs a vector of size embed_dim."""
    def __init__(self, embed_dim: int, n_qubits: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_qubits = n_qubits
        self.q_device = tq.QuantumDevice(n_wires=n_qubits, bsz=1)
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.linear = nn.Linear(n_qubits, embed_dim)

    def forward(self, x: torch.Tensor):
        B, T = x.shape[:2]
        outputs = []
        for t in range(T):
            idx = torch.tensor([t], device=x.device, dtype=torch.float32)
            qdev = self.q_device.copy(bsz=1, device=x.device)
            qdev.reset()
            self.encoder(qdev, idx)
            out = self.measure(qdev)
            outputs.append(out)
        out = torch.stack(outputs, dim=1)  # B,T,n_qubits
        out = self.linear(out)
        return out


class TextClassifierQuantum(nn.Module):
    """Quantum‑enhanced text classifier."""
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        n_qubits_attention: int = 8,
        n_qubits_ffn: int = 8,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_enc = PositionalEncodingQuantum(embed_dim, n_qubits_attention)
        self.blocks = nn.ModuleList(
            [
                MultiHeadAttentionQuantum(embed_dim, num_heads, dropout)
                for _ in range(num_blocks)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor):
        x = self.token_emb(x)
        x = self.pos_enc(x)
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)


__all__ = [
    "MultiHeadAttentionQuantum",
    "FeedForwardQuantum",
    "PositionalEncodingQuantum",
    "TextClassifierQuantum",
]
