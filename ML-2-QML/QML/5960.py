import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

class QuantumSelfAttention:
    """Quantum circuit implementing a self‑attention style block."""
    def __init__(self, n_qubits: int, shots: int = 1024):
        self.n_qubits = n_qubits
        self.shots = shots
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")
        self.backend = qiskit.Aer.get_backend("qasm_simulator")

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

    def run(self, inputs: np.ndarray) -> np.ndarray:
        rotation_params = np.random.uniform(0, 2 * np.pi, 3 * self.n_qubits)
        entangle_params = np.random.uniform(0, 2 * np.pi, self.n_qubits - 1)
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = qiskit.execute(circuit, self.backend, shots=self.shots)
        counts = job.result().get_counts(circuit)
        probs = np.zeros(self.n_qubits)
        for bitstring, cnt in counts.items():
            bits = np.array([int(b) for b in bitstring[::-1]])
            probs += cnt * bits
        probs /= self.shots
        return probs

class QuantumSelfAttentionLayer(nn.Module):
    """Wraps QuantumSelfAttention to produce an attention vector per token."""
    def __init__(self, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.attn = QuantumSelfAttention(embed_dim, shots=512)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        out = []
        for i in range(batch):
            token_out = []
            for j in range(seq_len):
                vec = self.attn.run(x[i, j].detach().cpu().numpy())
                token_out.append(vec)
            out.append(torch.tensor(token_out, device=x.device, dtype=torch.float32))
        out = torch.stack(out)
        return self.dropout(out)

class FeedForwardClassical(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

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
        return x + self.pe[:, : x.size(1)]

class TextClassifier(nn.Module):
    """Transformer‑style classifier that uses a quantum self‑attention layer."""
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional = PositionalEncoder(embed_dim)
        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "attn": QuantumSelfAttentionLayer(embed_dim, dropout),
                        "ffn": FeedForwardClassical(embed_dim, ffn_dim, dropout),
                        "norm1": nn.LayerNorm(embed_dim),
                        "norm2": nn.LayerNorm(embed_dim),
                        "drop": nn.Dropout(dropout),
                    }
                )
                for _ in range(num_blocks)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.positional(tokens)
        for blk in self.blocks:
            attn_out = blk["attn"](x)
            x = blk["norm1"](x + blk["drop"](attn_out))
            ffn_out = blk["ffn"](x)
            x = blk["norm2"](x + blk["drop"](ffn_out))
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)

__all__ = [
    "QuantumSelfAttention",
    "QuantumSelfAttentionLayer",
    "FeedForwardClassical",
    "PositionalEncoder",
    "TextClassifier",
]
