"""SelfAttentionV2: Classical multi‑head attention block.

This class implements a multi‑head attention block that can be used as a drop‑in
replacement for the original SelfAttention.  It is fully differentiable,
supports training with a standard PyTorch optimizer, and exposes a ``train``
helper that runs a simple epoch‑based training loop.

The API is intentionally compatible with the original seed: ``run`` accepts
``rotation_params`` and ``entangle_params`` arguments which are ignored so that
existing code can be run unchanged.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionV2(nn.Module):
    """Classical multi‑head attention block."""

    def __init__(self, embed_dim: int, num_heads: int = 1, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input of shape (batch, seq_len, embed_dim).

        Returns
        -------
        torch.Tensor
            Output of shape (batch, seq_len, embed_dim).
        """
        batch, seq_len, _ = x.size()

        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)
        output = self.out_proj(attn_output)
        return output

    def run(self, inputs: torch.Tensor, rotation_params=None, entangle_params=None) -> torch.Tensor:
        """
        Compatibility wrapper that forwards to ``forward``.
        The ``rotation_params`` and ``entangle_params`` arguments are ignored.
        """
        return self.forward(inputs)

    def train_loop(self, dataloader, loss_fn, optimizer, epochs: int = 10,
                   device: torch.device | str = "cpu") -> None:
        """
        Simple training loop that updates the attention parameters.

        Parameters
        ----------
        dataloader : Iterable
            Iterable yielding (inputs, targets) batches.
        loss_fn : callable
            Loss function that accepts (outputs, targets).
        optimizer : torch.optim.Optimizer
            Optimizer for the module parameters.
        epochs : int
            Number of epochs to run.
        device : torch.device | str
            Device on which to perform the computation.
        """
        self.to(device)
        for epoch in range(epochs):
            self.train()
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = self.forward(inputs)
                loss = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()
            self.eval()
