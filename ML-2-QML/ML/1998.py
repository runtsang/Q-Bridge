from __future__ import annotations

import math
import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# 0.  Utility helpers
# --------------------------------------------------------------------------- #
def _log_gradients(module: nn.Module) -> None:
    """Log the norm of gradients for each parameter – useful for debugging."""
    for name, param in module.named_parameters():
        if param.grad is not None:
            logging.debug(f"Grad norm for {name}: {param.grad.norm():.4f}")


# --------------------------------------------------------------------------- #
# 1.  Core building blocks
# --------------------------------------------------------------------------- #
class MultiHeadAttentionBase(nn.Module):
    """Base class for attention mechanisms – keeps the bookkeeping for a hybrid training loop."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, bias: bool = False) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError(f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.bias = bias

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    """Standard multi‑head attention implemented using torch.nn.MultiheadAttention."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, bias: bool = False):
        super().__init__(embed_dim, num_heads, dropout, bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, bias=bias, batch_first=True)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        attn_output, _ = self.attn(q, k, v, key_padding_mask=mask)
        return self.out_proj(attn_output)


# --------------------------------------------------------------------------- #
# 2.  Feed‑forward / VAE
# --------------------------------------------------------------------------- #
class FeedForwardBase(nn.Module):
    """Base for the feed‑forward part of a transformer block."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class FeedForwardClassical(FeedForwardBase):
    """Standard two‑layer MLP."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class FeedForwardVAE(FeedForwardBase):
    """Replaces the feed‑forward MLP with a lightweight VAE that regularises the token representations.  The latent mean is forwarded to the next sub‑module."""
    def __init__(self, embed_dim: int, ffn_dim: int, latent_dim: int, dropout: float = 0.1):
        super().__init__(embed_dim, ffn_dim, dropout)
        self.vae = SimpleVAE(embed_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        recon, mu, _ = self.vae(x)
        # use the latent mean as the output representation
        return mu


class SimpleVAE(nn.Module):
    """A minimal VAE suitable for the transformer feed‑forward slot."""
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        self.fc2 = nn.Linear(latent_dim, input_dim)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        recon = torch.sigmoid(self.fc2(z))
        return recon, mu, logvar


# --------------------------------------------------------------------------- #
# 3.  Transformer block
# --------------------------------------------------------------------------- #
class TransformerBlockBase(nn.Module):
    """Shared transformer block infrastructure."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class TransformerBlockClassical(TransformerBlockBase):
    """Standard transformer block with classical attention and feed‑forward."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 dropout: float = 0.1, use_vae: bool = False, latent_dim: int = 16):
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        if use_vae:
            self.ffn = FeedForwardVAE(embed_dim, ffn_dim, latent_dim, dropout)
        else:
            self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


# --------------------------------------------------------------------------- #
# 4.  Positional encoding
# --------------------------------------------------------------------------- #
class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding as in the original transformer."""
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


# --------------------------------------------------------------------------- #
# 5.  Full model
# --------------------------------------------------------------------------- #
class QTransformerTorchGen(nn.Module):
    """Hybrid transformer that can optionally use a VAE feed‑forward
    and supports a quantum‑aware loss term during training."""
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        use_vae: bool = False,
        latent_dim: int = 16,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.blocks = nn.ModuleList(
            [TransformerBlockClassical(embed_dim, num_heads, ffn_dim,
                                       dropout, use_vae, latent_dim)
             for _ in range(num_blocks)]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    # --------------------------------------------------------------------- #
    # 5.1 Forward pass
    # --------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(x)
        x = self.pos_encoder(x)
        for block in self.blocks:
            x = block(x)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)

    # --------------------------------------------------------------------- #
    # 5.2 Hybrid training step
    # --------------------------------------------------------------------- #
    def train_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        device: torch.device,
        log_gradients: bool = False,
        fidelity_weight: float = 0.0,
    ) -> dict[str, float]:
        """
        One training iteration that:

        * moves data to ``device``
        * performs a forward pass
        * computes the loss + optional quantum‑fidelity term
        * back‑propagates, steps the optimizer
        * logs gradient norms if requested
        """
        self.train()
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = self(inputs)
        loss = loss_fn(outputs.squeeze(-1) if outputs.ndim == 2 else outputs, targets)

        # If a quantum‑aware loss is requested, add a simple fidelity term
        if fidelity_weight > 0.0:
            # Treat the raw logits as a probability distribution
            probs = torch.softmax(outputs, dim=-1).clamp(min=1e-7)
            # One‑hot target
            target_onehot = torch.nn.functional.one_hot(targets,
                                                        num_classes=outputs.shape[-1]).float()
            # Cosine similarity as a proxy for fidelity
            fidelity = torch.nn.functional.cosine_similarity(probs, target_onehot, dim=-1).mean()
            loss = loss + fidelity_weight * (1.0 - fidelity)

        loss.backward()
        if log_gradients:
            _log_gradients(self)
        optimizer.step()

        return {"loss": loss.item(), "fidelity": fidelity.item() if fidelity_weight > 0 else None}

    # --------------------------------------------------------------------- #
    # 5.3 Utility for multi‑device parallelism
    # --------------------------------------------------------------------- #
    def to_device(self, device: torch.device) -> None:
        """Convenient wrapper that moves the model to ``device``."""
        self.to(device)

__all__ = ["QTransformerTorchGen"]
