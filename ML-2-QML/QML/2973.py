import math
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
import torchquantum as tq
import torchquantum.functional as tqf

class QuantumSelfAttention:
    """Quantum self‑attention circuit implemented with Qiskit."""
    def __init__(self, n_qubits: int = 4):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, 'q')
        self.cr = ClassicalRegister(n_qubits, 'c')
        self.backend = Aer.get_backend('qasm_simulator')

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

    def run(self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            shots: int = 1024) -> dict:
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = execute(circuit, self.backend, shots=shots)
        return job.result().get_counts(circuit)

def SelfAttention():
    """Return an instance of the quantum self‑attention class."""
    return QuantumSelfAttention(n_qubits=4)

class QuantumFeedForward(tq.QuantumModule):
    """Simple quantum feed‑forward network using a small circuit."""
    def __init__(self, n_qubits: int, ffn_dim: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.ffn_dim = ffn_dim
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
        )
        self.parameters = nn.ModuleList([tq.RY(has_params=True, trainable=True) for _ in range(n_qubits)])
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, n_qubits)

    def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice):
        self.encoder(q_device, x)
        for wire, gate in enumerate(self.parameters):
            gate(q_device, wires=wire)
        out = self.measure(q_device)
        out = self.linear1(out)
        return self.linear2(F.relu(out))

class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor):
        return x + self.pe[:, :x.size(1)]

class QuantumTransformerBlock(nn.Module):
    """Transformer block that uses a quantum self‑attention and optional quantum feed‑forward."""
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 ffn_dim: int,
                 n_qubits_attn: int = 0,
                 n_qubits_ffn: int = 0,
                 dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.use_quantum_attn = n_qubits_attn > 0
        self.use_quantum_ffn = n_qubits_ffn > 0

        if self.use_quantum_attn:
            self.attn = QuantumSelfAttention(n_qubits=n_qubits_attn)
        else:
            self.attn = None

        if self.use_quantum_ffn:
            self.ffn = QuantumFeedForward(n_qubits=n_qubits_ffn, ffn_dim=ffn_dim)
        else:
            self.ffn = None

        self.classical_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.classical_ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, embed_dim)
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        attn_out, _ = self.classical_attn(x, x, x, key_padding_mask=mask)
        if self.use_quantum_attn:
            rotation_params = np.random.rand(3 * x.size(1))
            entangle_params = np.random.rand(x.size(1) - 1)
            q_counts = self.attn.run(rotation_params, entangle_params)
            probs = torch.tensor([q_counts.get(k, 0) for k in sorted(q_counts)], dtype=torch.float32)
            probs = probs / probs.sum() if probs.sum() > 0 else probs
            q_attn = probs.unsqueeze(0).expand_as(attn_out)
            attn_out = 0.5 * attn_out + 0.5 * q_attn
        x = self.norm1(x + self.dropout(attn_out))

        ffn_out = self.classical_ffn(x)
        if self.use_quantum_ffn:
            batch, seq, _ = x.shape
            outputs = []
            for i in range(seq):
                qdev = tq.QuantumDevice(n_wires=self.ffn.n_qubits, bsz=1, device=x.device)
                out = self.ffn(x[:, i, :], qdev)
                outputs.append(out)
            q_ffn = torch.stack(outputs, dim=1)
            ffn_out = 0.5 * ffn_out + 0.5 * q_ffn
        x = self.norm2(x + self.dropout(ffn_out))
        return x

class TextClassifier(nn.Module):
    """Hybrid text classifier that can switch between classical and quantum components."""
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int,
                 dropout: float = 0.1,
                 use_quantum_attention: bool = False,
                 use_quantum_ffn: bool = False,
                 n_qubits_attn: int = 0,
                 n_qubits_ffn: int = 0):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        blocks = []
        for _ in range(num_blocks):
            if use_quantum_attention or use_quantum_ffn:
                blocks.append(
                    QuantumTransformerBlock(
                        embed_dim,
                        num_heads,
                        ffn_dim,
                        n_qubits_attn=n_qubits_attn,
                        n_qubits_ffn=n_qubits_ffn,
                        dropout=dropout
                    )
                )
            else:
                blocks.append(
                    TransformerBlock(
                        embed_dim,
                        num_heads,
                        ffn_dim,
                        dropout=dropout
                    )
                )
        self.transformers = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor):
        x = self.token_embedding(x)
        x = self.pos_encoder(x)
        x = self.transformers(x)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)

__all__ = [
    "QuantumSelfAttention",
    "SelfAttention",
    "QuantumFeedForward",
    "QuantumTransformerBlock",
    "TextClassifier",
]
