"""Unified classical kernel and transformer architecture.

This module extends the original RBF kernel utilities and transformer
blocks to allow a hybrid pipeline that can optionally include quantum
sub‑modules.  The public API remains backward compatible with the
original `QuantumKernelMethod.py` while adding a `UnifiedKernelTransformer`
class that stitches together a kernel (classical or quantum) and a
transformer (classical or quantum).  The design is intentionally
lightweight – the quantum parts are supplied via a callback or a
dedicated quantum module (see the QML counterpart).

"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Sequence, Optional, Callable, Any

# --------------------------------------------------------------------------- #
# 1.  Classic RBF kernel utilities (kept for backward compatibility)
# --------------------------------------------------------------------------- #

class KernalAnsatz(nn.Module):
    """Legacy RBF kernel implementation – kept for API compatibility."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    """Wraps :class:`KernalAnsatz` to expose a 2‑D kernel matrix."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """Legacy helper that builds the Gram matrix from two lists of tensors."""
    kernel = Kernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# --------------------------------------------------------------------------- #
# 2.  Hybrid kernel – can delegate to an external (quantum) kernel function
# --------------------------------------------------------------------------- #

class HybridKernel(nn.Module):
    """
    A kernel that can use either the classical RBF implementation or a
    user‑supplied callable (e.g. a quantum kernel).
    """
    def __init__(self, gamma: float = 1.0, quantum_callable: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.quantum_callable = quantum_callable
        self.classical_kernel = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.quantum_callable is not None:
            return self.quantum_callable(x, y)
        return self.classical_kernel(x, y)

    def gram(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """Return a Gram matrix using the selected kernel."""
        return np.array([[self.forward(x, y).item() for y in b] for x in a])

# --------------------------------------------------------------------------- #
# 3.  Hybrid transformer – can replace attention and feed‑forward heads with
#    quantum sub‑modules via callbacks
# --------------------------------------------------------------------------- #

class _TransformerBlock(nn.Module):
    """
    Internal transformer block that accepts pre‑constructed attention and
    feed‑forward modules.  This keeps the public API simple while still
    allowing quantum sub‑modules.
    """
    def __init__(self, embed_dim: int, attn: nn.Module, ffn: nn.Module, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = attn
        self.ffn = ffn
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(self.attn, nn.MultiheadAttention):
            attn_out, _ = self.attn(x, x, x)
        else:
            attn_out = self.attn(x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class HybridTransformer(nn.Module):
    """
    A transformer encoder that can optionally replace its attention and
    feed‑forward sub‑modules with quantum implementations supplied as
    callables.  The default implementation is fully classical.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        num_blocks: int,
        dropout: float = 0.1,
        quantum_attention: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        quantum_ffn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            attn = quantum_attention if quantum_attention is not None else nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
            ffn = quantum_ffn if quantum_ffn is not None else nn.Sequential(
                nn.Linear(embed_dim, ffn_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(ffn_dim, embed_dim),
            )
            block = _TransformerBlock(embed_dim, attn, ffn, dropout)
            self.blocks.append(block)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.dropout(x.mean(dim=1))

# --------------------------------------------------------------------------- #
# 4.  Unified kernel‑transformer classifier
# --------------------------------------------------------------------------- #

class UnifiedKernelTransformer(nn.Module):
    """
    Combines a kernel (classical or quantum) with a transformer encoder
    to produce a classifier.  The kernel is applied to the input data to
    generate a feature matrix; this matrix is then passed through the
    transformer.  The final pooled representation is fed to a linear
    classifier.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the raw input vectors.
    embed_dim : int
        Embedding dimension used by the transformer.
    num_heads : int
        Number of attention heads.
    ffn_dim : int
        Dimensionality of the feed‑forward sub‑network.
    num_blocks : int
        Number of transformer blocks.
    num_classes : int
        Number of output classes.
    gamma : float, optional
        Kernel width (used only if a classical kernel is selected).
    use_quantum_kernel : bool, optional
        If True, a quantum kernel callable must be provided via
        ``quantum_kernel``.
    quantum_kernel : Callable[[torch.Tensor, torch.Tensor], torch.Tensor], optional
        A callable that returns a kernel value between two tensors.  It
        should be compatible with the signature of :class:`HybridKernel`.
    use_quantum_transformer : bool, optional
        If True, quantum attention and/or feed‑forward callables must be
        supplied via ``quantum_attention`` and ``quantum_ffn``.
    quantum_attention : Callable[[torch.Tensor, torch.Tensor], torch.Tensor], optional
        Quantum attention callable.
    quantum_ffn : Callable[[torch.Tensor], torch.Tensor], optional
        Quantum feed‑forward callable.
    """
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        num_blocks: int,
        num_classes: int,
        gamma: float = 1.0,
        use_quantum_kernel: bool = False,
        quantum_kernel: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        use_quantum_transformer: bool = False,
        quantum_attention: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        quantum_ffn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        self.kernel = HybridKernel(gamma, quantum_kernel if use_quantum_kernel else None)
        self.proj = nn.Linear(1, embed_dim)
        self.transformer = HybridTransformer(
            embed_dim,
            num_heads,
            ffn_dim,
            num_blocks,
            dropout=0.1,
            quantum_attention=quantum_attention if use_quantum_transformer else None,
            quantum_ffn=quantum_ffn if use_quantum_transformer else None,
        )
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim).

        Returns
        -------
        logits : torch.Tensor
            Classifier logits of shape (batch_size, num_classes).
        """
        # Compute kernel matrix for each sample against the batch itself
        gram = torch.tensor(self.kernel.gram(x, x), dtype=torch.float32, device=x.device)
        # Reshape to (batch_size, seq_len, embed_dim=1)
        seq = gram.unsqueeze(-1)
        # Project to the desired embedding dimension
        x = self.proj(seq)
        # Pass through transformer
        h = self.transformer(x)
        # Classifier
        logits = self.classifier(h)
        return logits

__all__ = [
    "KernalAnsatz",
    "Kernel",
    "kernel_matrix",
    "HybridKernel",
    "HybridTransformer",
    "UnifiedKernelTransformer",
]
