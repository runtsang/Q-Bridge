"""Enhanced classical classifier with residual connections, dropout and optional attention.

The class mirrors the quantum helper interface: `build_classifier_circuit` in the seed
is replaced by the constructor and a `metadata` property that exposes the same tuple
structure used by the quantum implementation.  This enables downstream experiments
to swap the two back‑ends without changing the training loop.

Key extensions:
* Residual blocks (input → block → output + input) increase gradient flow.
* Dropout regularises the network and improves generalisation on small data sets.
* Optional self‑attention module captures feature interactions that a pure feed‑forward
  network would miss.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantumClassifier(nn.Module):
    """
    Classical neural network that mimics the interface of the quantum
    `build_classifier_circuit` helper.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input feature vector.
    depth : int
        Number of residual blocks.
    dropout : float, default 0.1
        Dropout probability applied after each block.
    use_attention : bool, default False
        If True, a simple multi‑head self‑attention layer is inserted after
        every residual block.
    """

    def __init__(
        self,
        num_features: int,
        depth: int,
        dropout: float = 0.1,
        use_attention: bool = False,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.depth = depth
        self.dropout = dropout
        self.use_attention = use_attention

        blocks: List[nn.Module] = []
        in_dim = num_features
        for _ in range(depth):
            linear = nn.Linear(in_dim, num_features)
            block = nn.Sequential(
                linear,
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            if use_attention:
                # simple multi‑head attention (1 head, query/key/value same as input)
                attn = nn.MultiheadAttention(
                    embed_dim=num_features,
                    num_heads=1,
                    batch_first=True,
                )
                block.add_module("attention", attn)
            blocks.append(block)
            in_dim = num_features

        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Linear(num_features, 2)

        # expose metadata similar to the quantum implementation
        self.encoding: List[int] = list(range(num_features))
        self.weight_sizes: List[int] = [
            sum(p.numel() for p in block.parameters())
            for block in self.blocks
        ] + [self.head.weight.numel() + self.head.bias.numel()]
        self.observables: List[int] = [0, 1]  # class indices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional residual connections."""
        out = x
        for block in self.blocks:
            residual = out
            out = block(out)
            if self.use_attention:
                # attention expects (batch, seq, embed)
                out, _ = block.attention(out.unsqueeze(1), out.unsqueeze(1), out.unsqueeze(1))
                out = out.squeeze(1)
            out = out + residual  # residual connection
        logits = self.head(out)
        return logits

    def metadata(self) -> Tuple[Iterable[int], Iterable[int], List[int]]:
        """Return (encoding, weight_sizes, observables) for compatibility."""
        return self.encoding, self.weight_sizes, self.observables

    # ------------------------------------------------------------------
    # Simple training helpers – not part of the core API but useful for demos
    # ------------------------------------------------------------------
    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        epochs: int = 100,
        lr: float = 1e-3,
        verbose: bool = False,
    ) -> None:
        """Train the network using cross‑entropy loss."""
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        for epoch in range(epochs):
            logits = self.forward(X)
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if verbose and (epoch + 1) % 10 == 0:
                acc = (logits.argmax(dim=1) == y).float().mean().item()
                print(f"Epoch {epoch+1:03d} | loss={loss.item():.4f} | acc={acc:.4f}")

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Return class predictions."""
        with torch.no_grad():
            logits = self.forward(X)
        return logits.argmax(dim=1)


__all__ = ["QuantumClassifier"]
