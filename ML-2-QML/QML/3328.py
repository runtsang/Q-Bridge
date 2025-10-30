"""Hybrid kernel + transformer – quantum implementation.

This module defines :class:`QuantumKernelTransformer` that uses a
quantum‑encoded RBF kernel (via TorchQuantum) and a transformer
classifier whose attention and feed‑forward layers are realized as
quantum circuits.  The public API matches the classical counterpart
so that the same class name can be used interchangeably.
"""

import math
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchquantum as tq
from torchquantum.functional import func_name_dict

# --------------------------------------------------------------------------- #
# 1. Quantum RBF kernel
# --------------------------------------------------------------------------- #
class QuantumKernel(nn.Module):
    """Quantum‑encoded RBF kernel using TorchQuantum."""

    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(self.n_wires)]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return |<psi(x)|psi(y)>|^2 using the overlap circuit."""
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.q_device.reset_states(self.q_device.bsz)
        # encode x
        self.encoder(self.q_device, x)
        # reversed encoding of y for overlap
        for info in reversed(self.encoder._func_list):
            params = y[:, info["input_idx"]] if info["num_params"] else None
            if params is not None:
                func_name_dict[info["func"]](self.q_device, wires=info["wires"], params=params)
            else:
                func_name_dict[info["func"]](self.q_device, wires=info["wires"])
        return torch.abs(self.q_device.states.view(-1)[0])

def kernel_matrix(a: Iterable[torch.Tensor], b: Iterable[torch.Tensor]) -> np.ndarray:
    """Return the Gram matrix using the quantum RBF kernel."""
    kernel = QuantumKernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# --------------------------------------------------------------------------- #
# 2. Quantum transformer blocks
# --------------------------------------------------------------------------- #
class QuantumAttention(nn.Module):
    """Multi‑head attention where each head is a small quantum circuit."""

    class _QLayer(tq.QuantumModule):
        def __init__(self, num_wires: int) -> None:
            super().__init__()
            self.n_wires = num_wires
            self.enc = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(self.n_wires)]
            )
            self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(self.n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.enc(qdev, x)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
                if wire < self.n_wires - 1:
                    tq.cnot(qdev, wires=[wire, wire + 1])
                else:
                    tq.cnot(qdev, wires=[self.n_wires - 1, 0])
            return self.measure(qdev)

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1,
                 n_wires: int = 8, q_device: Optional[tq.QuantumDevice] = None) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.q_layer = self._QLayer(n_wires)
        self.q_device = q_device or tq.QuantumDevice(n_wires=n_wires)
        self.combine = nn.Linear(embed_dim, embed_dim, bias=False)

    def _apply_head(self, x: torch.Tensor) -> torch.Tensor:
        """Project x into a quantum‑encoded vector per head."""
        heads = []
        for token in x.unbind(dim=1):
            token = token.view(token.size(0), self.num_heads, self.d_k)
            head_outputs = []
            for head in token.unbind(dim=1):
                qdev = self.q_device.copy(bsz=head.size(0), device=head.device)
                out = self.q_layer(head, qdev)
                head_outputs.append(out)
                self.q_device.reset_states()
                self.q_device = qdev
            heads.append(torch.stack(head_outputs, dim=1))
        return torch.stack(heads, dim=1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        k_proj = self._apply_head(x)
        q_proj = self._apply_head(x)
        v_proj = self._apply_head(x)
        k = k_proj.view(x.size(0), x.size(1), self.num_heads, self.d_k).transpose(1, 2)
        q = q_proj.view(x.size(0), x.size(1), self.num_heads, self.d_k).transpose(1, 2)
        v = v_proj.view(x.size(0), x.size(1), self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        out = torch.matmul(scores, v)
        out = out.transpose(1, 2).contiguous().view(x.size(0), x.size(1), self.embed_dim)
        return self.combine(out)

class QuantumFFN(nn.Module):
    """Feed‑forward network realized by a quantum module."""

    class _QLayer(tq.QuantumModule):
        def __init__(self, n_qubits: int) -> None:
            super().__init__()
            self.n_qubits = n_qubits
            self.enc = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
            )
            self.params = nn.ModuleList([tq.RY(has_params=True, trainable=True) for _ in range(n_qubits)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.enc(qdev, x)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            return self.measure(qdev)

    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.q_device = tq.QuantumDevice(n_qubits)
        self.q_layer = self._QLayer(n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim, bias=False)
        self.linear2 = nn.Linear(ffn_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for token in x.unbind(dim=1):
            qdev = self.q_device.copy(bsz=token.size(0), device=token.device)
            out = self.q_layer(token, qdev)
            outputs.append(out)
        out = torch.stack(outputs, dim=1)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))

class QuantumTransformerBlock(nn.Module):
    """Transformer block with quantum attention and feed‑forward."""

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 dropout: float = 0.1, n_wires: int = 8, n_qubits_ffn: int = 8) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = QuantumAttention(embed_dim, num_heads, dropout, n_wires)
        self.ffn = QuantumFFN(embed_dim, ffn_dim, n_qubits_ffn, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding (identical to classical)."""

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

class QuantumTextClassifier(nn.Module):
    """Transformer‑based text classifier with quantum sub‑modules."""

    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int,
                 dropout: float = 0.1,
                 n_wires: int = 8,
                 n_qubits_ffn: int = 8) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)
        self.transformers = nn.Sequential(
            *[QuantumTransformerBlock(embed_dim, num_heads, ffn_dim,
                                      dropout, n_wires, n_qubits_ffn)
              for _ in range(num_blocks)]
        )
        self.dropout = nn.Dropout(dropout)
        if num_classes > 2:
            self.classifier = nn.Linear(embed_dim, num_classes)
        else:
            self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_embedding(tokens)
        x = self.transformers(x)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)

# --------------------------------------------------------------------------- #
# 3. Hybrid class
# --------------------------------------------------------------------------- #
class QuantumKernelTransformer(nn.Module):
    """Hybrid kernel + transformer (quantum implementation).

    Parameters
    ----------
    vocab_size : int
        Size of the token vocabulary.
    embed_dim : int
        Embedding dimensionality.
    num_heads : int
        Number of attention heads.
    num_blocks : int
        Number of transformer blocks.
    ffn_dim : int
        Dimension of the feed‑forward network.
    num_classes : int
        Number of output classes.
    dropout : float, optional
        Dropout probability.
    n_wires : int, optional
        Number of qubits per attention head.
    n_qubits_ffn : int, optional
        Number of qubits in the feed‑forward quantum layer.
    """

    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int,
                 dropout: float = 0.1,
                 n_wires: int = 8,
                 n_qubits_ffn: int = 8) -> None:
        super().__init__()
        self.kernel = QuantumKernel()
        self.transformer = QuantumTextClassifier(vocab_size, embed_dim, num_heads,
                                                 num_blocks, ffn_dim, num_classes,
                                                 dropout, n_wires, n_qubits_ffn)

    def compute_kernel_matrix(self,
                              a: Iterable[torch.Tensor],
                              b: Iterable[torch.Tensor]) -> np.ndarray:
        """Compute Gram matrix using the quantum RBF kernel."""
        return np.array([[self.kernel(x, y).item() for y in b] for x in a])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the quantum transformer classifier."""
        return self.transformer(x)

__all__ = [
    "QuantumKernel",
    "kernel_matrix",
    "QuantumAttention",
    "QuantumFFN",
    "QuantumTransformerBlock",
    "PositionalEncoder",
    "QuantumTextClassifier",
    "QuantumKernelTransformer",
]
