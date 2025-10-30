"""Quantum‑only version of the UnifiedKernelTransformer.

This module implements the same public API as the classical
`UnifiedKernelTransformer` but replaces the kernel and transformer
sub‑modules with quantum implementations based on TorchQuantum.
The quantum kernel evaluates a variational circuit on two input
vectors and returns the absolute overlap of the resulting states.
The quantum transformer processes a sequence of token embeddings
through a quantum circuit that acts as a variational attention
and feed‑forward mechanism.

The code retains the legacy classes `KernalAnsatz`, `Kernel`,
`kernel_matrix` for backward compatibility, while exposing
`HybridKernel`, `HybridTransformer`, and `UnifiedKernelTransformer`
with quantum defaults.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
from torchquantum.functional import func_name_dict
from typing import Iterable, Sequence, Optional, Callable, Any

# --------------------------------------------------------------------------- #
# 1.  Quantum kernel utilities (kept for backward compatibility)
# --------------------------------------------------------------------------- #

class KernalAnsatzQ(tq.QuantumModule):
    """Quantum RBF kernel ansatz – identical to the seed but with a
    small modification: the circuit now includes a CNOT ladder to
    entangle all qubits, which improves expressivity."""
    def __init__(self, func_list):
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        # Encode x
        for info in self.func_list:
            params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        # Entangle all qubits
        for i in range(q_device.n_wires - 1):
            tq.cnot(q_device, wires=[i, i + 1])
        tq.cnot(q_device, wires=[q_device.n_wires - 1, 0])
        # Encode -y
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

class KernelQ(tq.QuantumModule):
    """Quantum kernel module that wraps :class:`KernalAnsatzQ`."""
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = KernalAnsatzQ(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

    def gram(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Compute the Gram matrix for two batches of tensors."""
        batch_a = a.size(0)
        batch_b = b.size(0)
        gram = torch.zeros(batch_a, batch_b, device=a.device)
        for i in range(batch_a):
            for j in range(batch_b):
                gram[i, j] = self.forward(a[i], b[j])
        return gram

def kernel_matrixQ(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    """Compute the Gram matrix using the quantum kernel."""
    kernel = KernelQ()
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# --------------------------------------------------------------------------- #
# 2.  Quantum attention and feed‑forward modules
# --------------------------------------------------------------------------- #

class QAttention(tq.QuantumModule):
    """Quantum attention that encodes the input sequence into a
    quantum state, applies a variational circuit, and measures the
    overlap with a reference state."""
    def __init__(self, n_wires: int, embed_dim: int):
        super().__init__()
        self.n_wires = n_wires
        self.embed_dim = embed_dim
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(n_wires)]
        )
        self.operators = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
        token = x[:, :self.n_wires, 0]
        self.encoder(q_device, token)
        for i, op in enumerate(self.operators):
            op(q_device, wires=i)
        for i in range(self.n_wires - 1):
            tq.cnot(q_device, wires=[i, i + 1])
        tq.cnot(q_device, wires=[self.n_wires - 1, 0])
        return self.measure(q_device)

class QFeedForward(tq.QuantumModule):
    """Quantum feed‑forward that maps a quantum state to a classical vector."""
    def __init__(self, n_wires: int, ffn_dim: int):
        super().__init__()
        self.n_wires = n_wires
        self.ffn_dim = ffn_dim
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(n_wires)]
        )
        self.operators = nn.ModuleList([tq.RY(has_params=True, trainable=True) for _ in range(n_wires)])
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.linear = nn.Linear(n_wires, ffn_dim)

    def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
        token = x[:, :self.n_wires, 0]
        self.encoder(q_device, token)
        for i, op in enumerate(self.operators):
            op(q_device, wires=i)
        out = self.measure(q_device)
        out = F.relu(out)
        return self.linear(out)

# --------------------------------------------------------------------------- #
# 3.  Quantum transformer block
# --------------------------------------------------------------------------- #

class QuantumTransformerBlock(nn.Module):
    """Transformer block that uses quantum attention and feed‑forward."""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        n_qubits_attn: int,
        n_qubits_ffn: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = QAttention(n_qubits_attn, embed_dim)
        self.ffn = QFeedForward(n_qubits_ffn, ffn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_device = tq.QuantumDevice(n_wires=self.attn.n_wires, bsz=x.size(0), device=x.device)
        attn_out = self.attn(x, q_device)
        x = self.norm1(x + self.dropout(attn_out))
        q_device = tq.QuantumDevice(n_wires=self.ffn.n_wires, bsz=x.size(0), device=x.device)
        ffn_out = self.ffn(x, q_device)
        return self.norm2(x + self.dropout(ffn_out))

# --------------------------------------------------------------------------- #
# 4.  Classical transformer block (for hybrid usage)
# --------------------------------------------------------------------------- #

class ClassicalTransformerBlock(nn.Module):
    """Standard transformer block implemented classically."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

# --------------------------------------------------------------------------- #
# 5.  Hybrid transformer – can mix classical and quantum blocks
# --------------------------------------------------------------------------- #

class HybridTransformerQ(nn.Module):
    """Hybrid transformer that can mix classical and quantum blocks."""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        num_blocks: int,
        dropout: float = 0.1,
        quantum_blocks: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            if quantum_blocks is not None and i < quantum_blocks:
                block = QuantumTransformerBlock(
                    embed_dim,
                    num_heads,
                    ffn_dim,
                    n_qubits_attn=embed_dim,
                    n_qubits_ffn=ffn_dim,
                    dropout=dropout,
                )
            else:
                block = ClassicalTransformerBlock(embed_dim, num_heads, ffn_dim, dropout)
            self.blocks.append(block)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.dropout(x.mean(dim=1))

# --------------------------------------------------------------------------- #
# 6.  Unified kernel‑transformer classifier (quantum version)
# --------------------------------------------------------------------------- #

class UnifiedKernelTransformerQ(nn.Module):
    """Quantum‑only version of the unified kernel‑transformer."""
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        num_blocks: int,
        num_classes: int,
        gamma: float = 1.0,
    ) -> None:
        super().__init__()
        self.kernel = KernelQ()
        self.proj = nn.Linear(1, embed_dim)
        self.transformer = HybridTransformerQ(
            embed_dim,
            num_heads,
            ffn_dim,
            num_blocks,
            dropout=0.1,
            quantum_blocks=num_blocks,
        )
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gram = self.kernel.gram(x, x)
        seq = gram.unsqueeze(-1)
        x = self.proj(seq)
        h = self.transformer(x)
        logits = self.classifier(h)
        return logits

# Alias to keep the public name consistent
class UnifiedKernelTransformer(UnifiedKernelTransformerQ):
    """Alias for quantum unified kernel transformer."""
    pass

__all__ = [
    "KernalAnsatzQ",
    "KernelQ",
    "kernel_matrixQ",
    "QAttention",
    "QFeedForward",
    "QuantumTransformerBlock",
    "HybridTransformerQ",
    "UnifiedKernelTransformerQ",
    "UnifiedKernelTransformer",
]
