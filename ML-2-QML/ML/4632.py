"""Hybrid Transformer Classifier combining classical and quantum modules.

The implementation mirrors the original QTransformerTorch API while adding optional
quantum sub‑modules (convolution, attention, feed‑forward).  The class is fully
compatible with the original anchor but can be instantiated with
`use_quantum_*` flags to switch to quantum behaviour.
"""

from __future__ import annotations

import math
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# 1. Convolutional front‑end – classical and quantum filters
# --------------------------------------------------------------------------- #

class _ClassicalConvFilter(nn.Module):
    """Simple 2‑D convolutional filter that emulates the quantum quanvolution."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        data : torch.Tensor
            Shape (batch, 1, H, W).  Only the first ``kernel_size``×``kernel_size`` patch
            is processed per forward call.

        Returns
        -------
        torch.Tensor
            Scalar activations per image, shape (batch, 1).
        """
        x = data[..., :self.kernel_size, :self.kernel_size]
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean(dim=[2, 3, 4]).unsqueeze(-1)

class _QuantumConvFilter:
    """Quantum analogue of the 2‑D convolutional filter using Qiskit."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0,
                 backend=None, shots: int = 100) -> None:
        import numpy as np
        import qiskit
        from qiskit.circuit.random import random_circuit

        self.n_qubits = kernel_size ** 2
        self.threshold = threshold
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots

        self.circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self.circuit.rx(self.theta[i], i)
        self.circuit.barrier()
        self.circuit += random_circuit(self.n_qubits, 2)
        self.circuit.measure_all()

    def run(self, data: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        data : torch.Tensor
            Shape (batch, 1, H, W).  Only the first ``kernel_size``×``kernel_size`` patch
            is processed per forward call.

        Returns
        -------
        torch.Tensor
            Scalar activations per image, shape (batch, 1).
        """
        import numpy as np
        from qiskit import execute

        batch = data.shape[0]
        patch = data[..., :self.circuit.num_qubits // int(math.sqrt(self.circuit.num_qubits)),
                        :self.circuit.num_qubits // int(math.sqrt(self.circuit.num_qubits))].reshape(batch, -1)
        counts = 0.0
        for sample in patch:
            param_binds = {theta: np.pi if val > self.threshold else 0 for theta, val in zip(self.theta, sample)}
            job = execute(self.circuit, self.backend, shots=self.shots, parameter_binds=[param_binds])
            result = job.result().get_counts(self.circuit)
            ones = sum(int(bit) for key, val in result.items() for bit in key)  # total number of |1>
            counts += ones / (self.shots * self.circuit.num_qubits)
        return torch.tensor(counts / batch, dtype=torch.float32).unsqueeze(-1)

class ConvModule(nn.Module):
    """Wrapper that selects classical or quantum convolution."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0,
                 use_quantum: bool = False, shots: int = 100):
        super().__init__()
        if use_quantum:
            self.conv = _QuantumConvFilter(kernel_size, threshold, shots=shots)
        else:
            self.conv = _ClassicalConvFilter(kernel_size, threshold)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        if isinstance(self.conv, _QuantumConvFilter):
            return self.conv.run(data)
        else:
            return self.conv(data)

# --------------------------------------------------------------------------- #
# 2. Transformer primitives – classical and quantum
# --------------------------------------------------------------------------- #

class MultiHeadAttentionBase(nn.Module):
    """Shared base for all attention variants."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Split last dimension into (num_heads, d_k) and transpose."""
        batch_size = x.size(0)
        return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError

class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    """Standard multi‑head attention implemented with PyTorch."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__(embed_dim, num_heads, dropout)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)

        q = self.separate_heads(q)
        k = self.separate_heads(k)
        v = self.separate_heads(v)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(x.size(0), -1, self.embed_dim)
        return self.out_proj(self.dropout(out))

class MultiHeadAttentionQuantum(MultiHeadAttentionBase):
    """Quantum‑enriched attention that projects queries, keys and values through a
    small variational circuit per head before classical attention."""
    class _QLayer(nn.Module):
        """One‑head quantum wrapper."""
        def __init__(self, n_wires: int = 8):
            super().__init__()
            import torchquantum as tq
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            self.parameters = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: 'tq.QuantumDevice') -> torch.Tensor:
            self.encoder(q_device, x)
            for gate in self.parameters:
                gate(q_device)
            return self.measure(q_device)

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, n_wires: int = 8):
        super().__init__(embed_dim, num_heads, dropout)
        self.n_wires = n_wires
        self.q_layer = self._QLayer(n_wires)
        self.combine = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch = x.size(0)
        # Classical linear projections
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)
        # Quantum head processing
        q = self.separate_heads(q)
        k = self.separate_heads(k)
        v = self.separate_heads(v)
        q = self._apply_qgate(q, batch)
        k = self._apply_qgate(k, batch)
        v = self._apply_qgate(v, batch)
        # Classical attention on quantum‑processed tensors
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, -1, self.embed_dim)
        return self.combine(self.dropout(out))

    def _apply_qgate(self, x: torch.Tensor, batch: int) -> torch.Tensor:
        import torchquantum as tq
        out = []
        for i in range(x.size(1)):  # per head
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=batch, device=x.device)
            out.append(self.q_layer(x[:, i, :], qdev))
        return torch.stack(out, dim=1)

class FeedForwardBase(nn.Module):
    """Base for classical and quantum feed‑forward."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class FeedForwardClassical(FeedForwardBase):
    """Standard two‑layer MLP."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class FeedForwardQuantum(FeedForwardBase):
    """Feed‑forward realised by a small quantum circuit."""
    class _QLayer(nn.Module):
        def __init__(self, n_qubits: int):
            super().__init__()
            import torchquantum as tq
            self.n_qubits = n_qubits
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
            )
            self.parameters = nn.ModuleList([tq.RY(has_params=True, trainable=True) for _ in range(n_qubits)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: 'tq.QuantumDevice') -> torch.Tensor:
            self.encoder(q_device, x)
            for gate in self.parameters:
                gate(q_device)
            return self.measure(q_device)

    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int, dropout: float = 0.1):
        super().__init__(embed_dim, ffn_dim, dropout)
        self.n_qubits = n_qubits
        self.q_layer = self._QLayer(n_qubits)
        self.q_device = tq.QuantumDevice(n_wires=n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = []
        for token in x.unbind(dim=1):
            qdev = self.q_device.copy(bsz=token.size(0), device=token.device)
            out.append(self.q_layer(token, qdev))
        out = torch.stack(out, dim=1)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))

class TransformerBlockBase(nn.Module):
    """Base transformer block containing attention and feed‑forward."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class TransformerBlockClassical(TransformerBlockBase):
    """Standard transformer block."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class TransformerBlockQuantum(TransformerBlockBase):
    """Transformer block that optionally inserts quantum attention or feed‑forward."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 use_quantum_attn: bool = False, use_quantum_ffn: bool = False,
                 q_n_wires: int = 8, q_ffn_qubits: int = 8, dropout: float = 0.1):
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = (MultiHeadAttentionQuantum(embed_dim, num_heads, dropout, q_n_wires)
                     if use_quantum_attn else
                     MultiHeadAttentionClassical(embed_dim, num_heads, dropout))
        self.ffn = (FeedForwardQuantum(embed_dim, ffn_dim, q_ffn_qubits, dropout)
                     if use_quantum_ffn else
                     FeedForwardClassical(embed_dim, ffn_dim, dropout))

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
        return x + self.pe[:, : x.size(1)]

class HybridTransformerClassifier(nn.Module):
    """Transformer‑based classifier that can interchange classical and quantum
    sub‑modules.  The model can process both tokenized text and image patches."""
    def __init__(self,
                 vocab_size: Optional[int] = None,
                 embed_dim: int = 128,
                 num_heads: int = 8,
                 num_blocks: int = 4,
                 ffn_dim: int = 512,
                 num_classes: int = 2,
                 dropout: float = 0.1,
                 conv_kernel_size: int = 2,
                 conv_threshold: float = 0.0,
                 use_quantum_conv: bool = False,
                 use_quantum_attn: bool = False,
                 use_quantum_ffn: bool = False,
                 q_n_wires: int = 8,
                 q_ffn_qubits: int = 8,
                 ):
        super().__init__()
        self.token_embedding = (nn.Embedding(vocab_size, embed_dim)
                                if vocab_size is not None else None)
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.conv_module = ConvModule(conv_kernel_size, conv_threshold,
                                      use_quantum=use_quantum_conv,
                                      shots=100)

        block_cls = TransformerBlockQuantum if (use_quantum_attn or use_quantum_ffn) else TransformerBlockClassical
        self.transformers = nn.ModuleList([
            block_cls(embed_dim, num_heads, ffn_dim,
                      use_quantum_attn=use_quantum_attn,
                      use_quantum_ffn=use_quantum_ffn,
                      q_n_wires=q_n_wires,
                      q_ffn_qubits=q_ffn_qubits,
                      dropout=dropout)
            for _ in range(num_blocks)
        ])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            * For text: LongTensor of shape (batch, seq_len) containing token indices.
            * For images: FloatTensor of shape (batch, 1, H, W) – grayscale.

        Returns
        -------
        torch.Tensor
            Class logits of shape (batch, num_classes).
        """
        if self.token_embedding is not None and x.ndim == 2:
            # Text input
            token_emb = self.token_embedding(x)          # (B, L, D)
            x = self.pos_encoder(token_emb)              # add positional encoding
        else:
            # Image input – use convolution to produce a scalar per patch
            conv_out = self.conv_module(x)               # (B, 1)
            # Broadcast to embedding dimension
            x = conv_out.repeat(1, self.pos_encoder.pe.size(1), 1).transpose(1, 2)

        for block in self.transformers:
            x = block(x)

        x = self.dropout(x.mean(dim=1))                  # global average pooling
        return self.classifier(x)

__all__ = [
    "ConvModule",
    "MultiHeadAttentionBase",
    "MultiHeadAttentionClassical",
    "MultiHeadAttentionQuantum",
    "FeedForwardBase",
    "FeedForwardClassical",
    "FeedForwardQuantum",
    "TransformerBlockBase",
    "TransformerBlockClassical",
    "TransformerBlockQuantum",
    "PositionalEncoder",
    "HybridTransformerClassifier",
]
