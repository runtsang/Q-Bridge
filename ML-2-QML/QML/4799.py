import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister


# ------------------------------------
# Quantum quanvolution components
# ------------------------------------
class QuanvolutionFilterQuantum(tq.QuantumModule):
    """Quantum kernel applied to 2×2 image patches."""
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.random_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                data = torch.stack([x[:, r, c], x[:, r, c + 1],
                                    x[:, r + 1, c], x[:, r + 1, c + 1]], dim=1)
                self.encoder(qdev, data)
                self.random_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)


class QuanvolutionClassifierQuantum(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.filter = QuanvolutionFilterQuantum()
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.filter(x)
        logits = self.linear(feats)
        return F.log_softmax(logits, dim=-1)


# ------------------------------------
# Quantum self‑attention helper
# ------------------------------------
class QuantumSelfAttention:
    """Self‑attention block realised on a Qiskit simulator."""
    def __init__(self, n_qubits: int = 4) -> None:
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _build_circuit(self, rot: np.ndarray, ent: np.ndarray) -> QuantumCircuit:
        circ = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            circ.rx(rot[3 * i], i)
            circ.ry(rot[3 * i + 1], i)
            circ.rz(rot[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circ.crx(ent[i], i, i + 1)
        circ.measure(self.qr, self.cr)
        return circ

    def run(self,
            backend,
            rot: np.ndarray,
            ent: np.ndarray,
            shots: int = 1024) -> dict:
        circ = self._build_circuit(rot, ent)
        job = qiskit.execute(circ, backend, shots=shots)
        return job.result().get_counts(circ)


# ------------------------------------
# Quantum transformer blocks
# ------------------------------------
class MultiHeadAttentionQuantum(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.k_lin = nn.Linear(embed_dim, embed_dim)
        self.q_lin = nn.Linear(embed_dim, embed_dim)
        self.v_lin = nn.Linear(embed_dim, embed_dim)
        self.combine = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, seq, _ = x.shape
        k = self.k_lin(x).view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        q = self.q_lin(x).view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_lin(x).view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch, seq, self.embed_dim)
        return self.combine(out)


class FeedForwardQuantum(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(n_qubits)]
        )
        self.rng = nn.ModuleList([tq.RY(has_params=True, trainable=True) for _ in range(n_qubits)])
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        out_tokens = []
        for i in range(seq):
            token = x[:, i, :]
            qdev = tq.QuantumDevice(self.n_qubits, bsz=batch, device=token.device)
            self.encoder(qdev, token)
            for gate in self.rng:
                gate(qdev)
            meas = self.measure(qdev)
            out_tokens.append(meas)
        out = torch.stack(out_tokens, dim=1)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))


class TransformerBlockQuantum(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 qnn_k: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, qnn_k, dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_out = self.attn(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


# ------------------------------------
# Positional encoding (shared)
# ------------------------------------
class PositionalEncoder(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


# ------------------------------------
# Quantum text classifier
# ------------------------------------
class TextClassifier(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int,
                 dropout: float = 0.1,
                 qnn_k: int = 8) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_enc = PositionalEncoder(embed_dim)
        self.transformer = nn.Sequential(
            *[TransformerBlockQuantum(embed_dim, num_heads, ffn_dim, qnn_k, dropout)
              for _ in range(num_blocks)]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_emb(x)
        x = self.pos_enc(tokens)
        x = self.transformer(x)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)


__all__ = [
    "QuanvolutionFilterQuantum",
    "QuanvolutionClassifierQuantum",
    "QuantumSelfAttention",
    "MultiHeadAttentionQuantum",
    "FeedForwardQuantum",
    "TransformerBlockQuantum",
    "PositionalEncoder",
    "TextClassifier",
]
