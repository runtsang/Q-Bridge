from __future__ import annotations

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
    """
    Quantum self‑attention that builds a parameterised circuit per token.
    The circuit consists of RX, RY, RZ rotations followed by a chain of
    controlled‑X gates.  The expectation value of Pauli‑Z on each wire is
    used as the attention output.
    """
    def __init__(self, n_qubits: int = 4):
        self.n_qubits = n_qubits
        self.backend = Aer.get_backend("qasm_simulator")

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> qiskit.QuantumCircuit:
        qr = QuantumRegister(self.n_qubits)
        cr = ClassicalRegister(self.n_qubits)
        circ = QuantumCircuit(qr, cr)
        for i in range(self.n_qubits):
            circ.rx(rotation_params[3 * i], i)
            circ.ry(rotation_params[3 * i + 1], i)
            circ.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circ.crx(entangle_params[i], i, i + 1)
        circ.measure(qr, cr)
        return circ

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
        shots: int = 1024,
    ) -> np.ndarray:
        """
        Execute the circuit for each token in the sequence and return the
        expectation values of Pauli‑Z on each qubit as a numpy array of shape
        (batch, seq, n_qubits).  The input tensor is ignored in this
        simplified implementation.
        """
        batch, seq, _ = inputs.shape
        outputs = []
        for _ in range(seq):
            circ = self._build_circuit(rotation_params, entangle_params)
            job = execute(circ, self.backend, shots=shots)
            result = job.result()
            counts = result.get_counts(circ)
            exp_vals = []
            for qubit in range(self.n_qubits):
                val = 0.0
                for state, cnt in counts.items():
                    bit = int(state[self.n_qubits - 1 - qubit])
                    val += (1 if bit == 0 else -1) * cnt
                val /= shots
                exp_vals.append(val)
            outputs.append(exp_vals)
        out = np.stack(outputs, axis=0)  # (seq, n_qubits)
        out = np.tile(out[np.newaxis, :, :], (batch, 1, 1))
        return out

class MultiHeadAttentionQuantum(tq.QuantumModule):
    """
    Multi‑head attention that replaces the linear projections with a small
    variational circuit per head.  The circuit is a 4‑qubit RX‑RY‑RZ chain
    followed by a small entangling pattern.  The output of the circuit is
    interpreted as a projection vector for the attention mechanism.
    """
    class _QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int = 4):
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [i], "func": "rx", "wires": [i]}
                    for i in range(n_wires)
                ]
            )
            self.params = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            for wire, gate in enumerate(self.params):
                gate(q_device, wires=wire)
            return self.measure(q_device)

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, q_device: Optional[tq.QuantumDevice] = None):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.q_layer = self._QLayer()
        self.q_device = q_device or tq.QuantumDevice(n_wires=self.q_layer.n_wires, bsz=1)
        self.combine_heads = nn.Linear(embed_dim, embed_dim, bias=False)

    def _apply_quantum(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the quantum layer to each head of each token.  The input is
        expected to be of shape (batch, seq, embed_dim).
        """
        batch, seq, _ = x.shape
        outputs = []
        for i in range(seq):
            token = x[:, i, :]  # (batch, embed_dim)
            head_out = []
            for _ in range(self.num_heads):
                qdev = self.q_device.copy(bsz=batch, device=token.device)
                head_out.append(self.q_layer(token, qdev))
            outputs.append(torch.stack(head_out, dim=1))
        return torch.stack(outputs, dim=1)  # (batch, seq, heads, wires)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        q = self._apply_quantum(x)
        k = self._apply_quantum(x)
        v = self._apply_quantum(x)

        # reshape heads
        batch, seq, heads, wires = q.shape
        q = q.view(batch, seq, heads * wires)
        k = k.view(batch, seq, heads * wires)
        v = v.view(batch, seq, heads * wires)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        scores = self.dropout(F.softmax(scores, dim=-1))
        attn = torch.matmul(scores, v)
        attn = attn.transpose(1, 2).contiguous().view(batch, seq, self.embed_dim)
        return self.combine_heads(attn)

class MultiHeadAttentionClassical(nn.Module):
    """
    Classical multi‑head attention used as a fallback when quantum heads are disabled.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq, _ = x.shape
        q = self.q_linear(x).view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(x).view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(x).view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        scores = self.dropout(F.softmax(scores, dim=-1))
        attn = torch.matmul(scores, v)
        attn = attn.transpose(1, 2).contiguous().view(batch, seq, self.embed_dim)
        return self.out_proj(attn)

class FeedForwardClassical(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim, bias=False)
        self.linear2 = nn.Linear(ffn_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class FeedForwardQuantum(tq.QuantumModule):
    """
    Feed‑forward network implemented as a small variational circuit that maps
    the token representation to a higher‑dimensional space and back.
    """
    class _QLayer(tq.QuantumModule):
        def __init__(self, n_qubits: int):
            super().__init__()
            self.n_qubits = n_qubits
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [i], "func": "rx", "wires": [i]}
                    for i in range(n_qubits)
                ]
            )
            self.params = nn.ModuleList(
                [tq.RY(has_params=True, trainable=True) for _ in range(n_qubits)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            for wire, gate in enumerate(self.params):
                gate(q_device, wires=wire)
            return self.measure(q_device)

    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.q_layer = self._QLayer(n_qubits)
        self.q_device = tq.QuantumDevice(n_wires=n_qubits, bsz=1)
        self.linear1 = nn.Linear(n_qubits, ffn_dim, bias=False)
        self.linear2 = nn.Linear(ffn_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        outputs = []
        for i in range(seq):
            token = x[:, i, :].view(batch, -1)
            qdev = self.q_device.copy(bsz=batch, device=token.device)
            q_out = self.q_layer(token, qdev)
            outputs.append(q_out)
        out = torch.stack(outputs, dim=1)  # (batch, seq, n_qubits)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))

class TransformerBlockQuantum(nn.Module):
    """
    Transformer block that optionally uses quantum attention and feed‑forward
    sub‑modules.  If the quantum sub‑modules are not enabled, classical
    counterparts are used.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        n_qubits_transformer: int = 0,
        n_qubits_ffn: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        if n_qubits_transformer > 0:
            self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout)
        else:
            self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)

        if n_qubits_ffn > 0:
            self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits_ffn)
        else:
            self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

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
        return x + self.pe[:, : x.size(1)]

class UnifiedSelfAttentionTransformer(nn.Module):
    """
    Quantum‑enhanced transformer that starts with a quantum self‑attention
    block and then applies a stack of transformer layers that can be either
    classical or quantum.  The class exposes the same API as its classical
    counterpart, enabling side‑by‑side experimentation.
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
        n_qubits_transformer: int = 0,
        n_qubits_ffn: int = 0,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)

        self.rotation_params = nn.Parameter(torch.randn(3 * embed_dim))
        self.entangle_params = nn.Parameter(torch.randn(embed_dim - 1))

        self.self_attention = QuantumSelfAttention(n_qubits=embed_dim)
        self.transformer = nn.Sequential(
            *[TransformerBlockQuantum(
                embed_dim,
                num_heads,
                ffn_dim,
                n_qubits_transformer,
                n_qubits_ffn,
                dropout=dropout,
            ) for _ in range(num_blocks)]
        )
        self.dropout = nn.Dropout(dropout)
        if num_classes > 2:
            self.classifier = nn.Linear(embed_dim, num_classes)
        else:
            self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_encoder(tokens)

        # Quantum self‑attention
        rotation_np = self.rotation_params.detach().cpu().numpy()
        entangle_np = self.entangle_params.detach().cpu().numpy()
        x_np = x.detach().cpu().numpy()
        sa_out = self.self_attention.run(rotation_np, entangle_np, x_np)
        x = torch.from_numpy(sa_out).to(x.device)

        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)

__all__ = [
    "QuantumSelfAttention",
    "MultiHeadAttentionQuantum",
    "FeedForwardQuantum",
    "TransformerBlockQuantum",
    "PositionalEncoder",
    "UnifiedSelfAttentionTransformer",
]
