"""QuantumHybridClassifier – quantum‑enhanced implementation.

The module implements a hybrid classifier that can operate as a
quantum variational circuit or as a transformer with quantum sub‑modules.
The class is compatible with TorchQuantum and Qiskit and can be
trained end‑to‑end with PyTorch optimizers.
"""

import math
from typing import Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

# Quantum‑enhanced transformer components
class MultiHeadAttentionQuantum(nn.Module):
    """Attention block that processes each head through a small
    variational circuit implemented with TorchQuantum.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, n_wires: int = 8):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)

        # Quantum sub‑module per head
        self.q_layers = nn.ModuleList(
            [self._build_q_layer(n_wires) for _ in range(num_heads)]
        )
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def _build_q_layer(self, n_wires: int):
        class QLayer(tq.QuantumModule):
            def __init__(self):
                super().__init__()
                self.n_wires = n_wires
                self.encoder = tq.GeneralEncoder(
                    [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
                )
                self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True)
                                             for _ in range(n_wires)])
                self.measure = tq.MeasureAll(tq.PauliZ)

            def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
                self.encoder(qdev, x)
                for w, gate in enumerate(self.params):
                    gate(qdev, wires=w)
                return self.measure(qdev)

        return QLayer()

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        # Linear projections
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)

        # Split heads
        q = q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Quantum projection for each head
        out_heads = []
        for head_idx in range(self.num_heads):
            head_q = q[:, head_idx]
            head_k = k[:, head_idx]
            head_v = v[:, head_idx]
            qdev = tq.QuantumDevice(n_wires=self.q_layers[head_idx].n_wires,
                                    bsz=head_q.size(0),
                                    device=head_q.device)
            head_out = self.q_layers[head_idx](head_q, qdev)
            out_heads.append(head_out.unsqueeze(1))
        out = torch.cat(out_heads, dim=1)  # (batch, heads, seq, d_k)

        # Attention scores
        scores = torch.matmul(out, out.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, out)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.out_proj(context)

class FeedForwardQuantum(nn.Module):
    """Feed‑forward sub‑module realised by a small variational circuit."""
    def __init__(self, embed_dim: int, ffn_dim: int, n_wires: int = 8, dropout: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.q_layer = self._build_q_layer(n_wires)
        self.linear1 = nn.Linear(n_wires, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def _build_q_layer(self, n_wires: int):
        class QLayer(tq.QuantumModule):
            def __init__(self):
                super().__init__()
                self.n_wires = n_wires
                self.encoder = tq.GeneralEncoder(
                    [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
                )
                self.params = nn.ModuleList([tq.RY(has_params=True, trainable=True)
                                             for _ in range(n_wires)])
                self.measure = tq.MeasureAll(tq.PauliZ)

            def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
                self.encoder(qdev, x)
                for w, gate in enumerate(self.params):
                    gate(qdev, wires=w)
                return self.measure(qdev)

        return QLayer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, embed_dim)
        q_out = []
        for token in x.unbind(dim=1):
            qdev = tq.QuantumDevice(n_wires=self.q_layer.n_wires,
                                    bsz=token.size(0),
                                    device=token.device)
            q_out.append(self.q_layer(token, qdev))
        q_out = torch.stack(q_out, dim=1)  # (batch, seq, n_wires)
        out = self.linear1(self.drop(q_out))
        return self.linear2(F.relu(out))

class TransformerBlockQuantum(nn.Module):
    """Transformer block that may use quantum attention and feed‑forward."""
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 ffn_dim: int,
                 n_wires_attn: int = 8,
                 n_wires_ffn: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout, n_wires_attn)
        self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_wires_ffn, dropout)
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
        div_term = torch.exp(torch.arange(0, embed_dim, 2) *
                             (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class QuantumHybridClassifier(nn.Module):
    """Hybrid transformer classifier that can toggle quantum sub‑modules.

    Parameters
    ----------
    vocab_size : int
        Size of the vocabulary (used for embedding lookup).
    embed_dim : int
        Embedding dimension.
    num_heads : int
        Number of attention heads.
    num_blocks : int
        Number of transformer blocks.
    ffn_dim : int
        Feed‑forward dimension.
    num_classes : int
        Number of output classes.
    use_quantum : bool
        If True, each block uses quantum attention and feed‑forward modules.
    n_wires_attn : int
        Number of quantum wires for the attention head circuit.
    n_wires_ffn : int
        Number of quantum wires for the feed‑forward circuit.
    """
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int,
                 use_quantum: bool = False,
                 n_wires_attn: int = 8,
                 n_wires_ffn: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        block_cls = TransformerBlockQuantum if use_quantum else TransformerBlockClassical
        self.transformer = nn.Sequential(
            *[block_cls(embed_dim, num_heads, ffn_dim,
                        n_wires_attn, n_wires_ffn, dropout)
              for _ in range(num_blocks)]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)

class TransformerBlockClassical(nn.Module):
    """Purely classical transformer block used when ``use_quantum=False``."""
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 ffn_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                          dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

# --------------------------------------------------------------------------- #
#  Quantum variational classifier (data‑uploading)
# --------------------------------------------------------------------------- #

def build_classifier_circuit(num_features: int,
                             depth: int,
                             n_qubits: int,
                             obs_type: str = "Z",
                             backend: str | None = None) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """
    Construct a Qiskit variational circuit that mirrors a classical
    feed‑forward classifier.  The circuit consists of an encoding layer
    followed by a stack of parameterised rotations and CNOT entanglement.
    It returns the circuit, the encoding and variational parameters,
    and the set of observables for measurement.

    Parameters
    ----------
    num_features : int
        Number of input features (must be <= n_qubits).
    depth : int
        Depth of the variational ansatz.
    n_qubits : int
        Number of qubits used for the circuit.
    obs_type : str, optional
        Observable type for measurement (currently only “Z” is supported).
    backend : str, optional
        Name of the Qiskit backend to optimise the circuit for.

    Returns
    -------
    circuit : QuantumCircuit
        The constructed circuit.
    encoding : list[ParameterVector]
        Parameters that encode the input data.
    variational : list[ParameterVector]
        Trainable parameters of the ansatz.
    observables : list[SparsePauliOp]
        Pauli‑Z operators used for readout.
    """
    if n_qubits < 2:
        raise ValueError("n_qubits must be at least 2 for a meaningful ansatz")

    encoding = ParameterVector("x", num_features)
    variational = ParameterVector("theta", num_features * depth)

    circuit = QuantumCircuit(n_qubits)
    # Data‑encoding with RX rotations
    for i, param in enumerate(encoding):
        circuit.rx(param, i)

    # Variational ansatz
    idx = 0
    for _ in range(depth):
        for q in range(n_qubits):
            circuit.ry(variational[idx], q)
            idx += 1
        # Entangling layer
        for q in range(n_qubits - 1):
            circuit.cz(q, q + 1)
        circuit.cz(n_qubits - 1, 0)

    # Measurement operators
    observables = [SparsePauliOp("I" * i + "Z" + "I" * (n_qubits - i - 1))
                   for i in range(n_qubits)]

    return circuit, list(encoding), list(variational), observables

__all__ = [
    "QuantumHybridClassifier",
    "TransformerBlockClassical",
    "TransformerBlockQuantum",
    "MultiHeadAttentionQuantum",
    "FeedForwardQuantum",
    "PositionalEncoder",
    "build_classifier_circuit",
]
