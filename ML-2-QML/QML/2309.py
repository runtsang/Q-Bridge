import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute

class QuantumSelfAttention(nn.Module):
    """Quantum self‑attention implemented with Qiskit."""
    def __init__(self, n_qubits: int = 4):
        super().__init__()
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")
        self.backend = Aer.get_backend("qasm_simulator")
        # Learnable parameters
        self.rotation_params = nn.Parameter(torch.randn(3 * n_qubits))
        self.entangle_params = nn.Parameter(torch.randn(n_qubits - 1))

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        circuit = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        circuit.measure(self.qr, self.cr)
        return circuit

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, shots: int = 1024) -> np.ndarray:
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = execute(circuit, self.backend, shots=shots)
        counts = job.result().get_counts(circuit)
        probs = np.zeros(2 ** self.n_qubits)
        for bitstring, cnt in counts.items():
            idx = int(bitstring[::-1], 2)
            probs[idx] = cnt
        probs /= probs.sum()
        return probs

class MultiHeadAttentionQuantumBase(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError

class MultiHeadAttentionQuantum(MultiHeadAttentionQuantumBase):
    """Quantum multi‑head attention using Qiskit circuits."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__(embed_dim, num_heads, dropout)
        self.heads = nn.ModuleList([QuantumSelfAttention() for _ in range(num_heads)])
        self.out_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        head_outs = []
        for head in self.heads:
            token_outs = []
            for i in range(batch * seq_len):
                token = x.view(-1, self.embed_dim)[i].detach().cpu().numpy()
                rotation = head.rotation_params.detach().cpu().numpy()
                entangle = head.entangle_params.detach().cpu().numpy()
                probs = head.run(rotation, entangle)
                token_outs.append(torch.from_numpy(probs).float())
            out = torch.stack(token_outs).view(batch, seq_len, -1)
            head_outs.append(out)
        out = torch.stack(head_outs, dim=1)  # (batch, heads, seq, dim)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)
        return self.out_linear(out)

class FeedForwardQuantum(nn.Module):
    """Quantum feed‑forward using a simple Qiskit circuit (placeholder)."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerBlockQuantum(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 dropout: float = 0.1, use_quantum_ffn: bool = False):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, dropout) if use_quantum_ffn else FeedForwardQuantum(embed_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class PositionalEncoder(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class HybridTransformer(nn.Module):
    """Transformer that optionally uses quantum attention and feed‑forward sub‑modules."""
    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int, num_blocks: int,
                 ffn_dim: int, num_classes: int, dropout: float = 0.1,
                 use_quantum_attention: bool = False, use_quantum_ffn: bool = False):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        block_cls = TransformerBlockQuantum if use_quantum_attention else TransformerBlockQuantum
        self.transformer_blocks = nn.Sequential(
            *[block_cls(embed_dim, num_heads, ffn_dim, dropout, use_quantum_ffn) for _ in range(num_blocks)]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_encoder(tokens)
        x = self.transformer_blocks(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)

__all__ = [
    "QuantumSelfAttention",
    "MultiHeadAttentionQuantum",
    "FeedForwardQuantum",
    "TransformerBlockQuantum",
    "PositionalEncoder",
    "HybridTransformer",
]
