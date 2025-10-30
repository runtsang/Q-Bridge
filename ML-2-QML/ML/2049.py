import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

class MultiHeadAttentionBase(nn.Module):
    """Base class for multi‑head attention."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError

class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    """Standard multi‑head attention."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_output, _ = self.attn(x, x, x, key_padding_mask=mask)
        return attn_output

class MultiHeadAttentionQuantum(MultiHeadAttentionClassical):
    """Placeholder for quantum attention – kept for API parity."""
    pass

class FeedForwardBase(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class FeedForwardClassical(FeedForwardBase):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class FeedForwardQuantum(FeedForwardClassical):
    """Placeholder for quantum feed‑forward – kept for API parity."""
    pass

class TransformerBlockBase(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class TransformerBlockClassical(TransformerBlockBase):
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
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, dropout)

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
        return x + self.pe[:, :x.size(1)]

class QTransformerTorchGen256(nn.Module):
    """
    Encoder‑decoder transformer that supports both classification and causal language modelling.
    The encoder can optionally use quantum sub‑modules; the decoder remains classical.
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
        max_len: int = 256,
        n_qubits_transformer: int = 0,
        n_qubits_ffn: int = 0,
        q_device=None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        # Encoder
        self.encoder_embed = nn.Embedding(vocab_size, embed_dim)
        self.encoder_pos = PositionalEncoder(embed_dim)
        encoder_blocks = []
        for _ in range(num_blocks):
            if n_qubits_transformer > 0:
                encoder_blocks.append(
                    TransformerBlockQuantum(
                        embed_dim, num_heads, ffn_dim, dropout
                    )
                )
            else:
                encoder_blocks.append(
                    TransformerBlockClassical(
                        embed_dim, num_heads, ffn_dim, dropout
                    )
                )
        self.encoder_blocks = nn.Sequential(*encoder_blocks)
        self.encoder_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

        # Decoder (classical)
        self.decoder_embed = nn.Embedding(vocab_size, embed_dim)
        self.decoder_pos = PositionalEncoder(embed_dim)
        decoder_blocks = []
        for _ in range(num_blocks):
            decoder_blocks.append(
                TransformerBlockClassical(
                    embed_dim, num_heads, ffn_dim, dropout
                )
            )
        self.decoder_blocks = nn.Sequential(*decoder_blocks)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def encode(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.encoder_embed(x)
        x = self.encoder_pos(x)
        x = self.encoder_blocks(x)
        return x

    def classify(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.encode(x, mask)
        x = self.encoder_dropout(x.mean(dim=1))
        return self.classifier(x)

    def decode(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.decoder_embed(x)
        x = self.decoder_pos(x)
        x = self.decoder_blocks(x)
        return self.lm_head(x)

    def forward(self, x: torch.Tensor, task: str = 'classify', mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if task == 'classify':
            return self.classify(x, mask)
        elif task == 'decode':
            return self.decode(x, mask)
        else:
            raise ValueError("task must be 'classify' or 'decode'")

    def generate(
        self,
        start_token: torch.Tensor,
        max_length: int = 256,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
    ) -> torch.Tensor:
        """
        Autoregressive generation from the decoder.
        :param start_token: Tensor of shape (batch, seq_len) with initial tokens
        :param max_length: maximum sequence length to generate
        :param temperature: sampling temperature
        :param top_k: keep only top_k tokens
        :param top_p: keep tokens with cumulative probability <= top_p
        :return: Tensor (batch, generated_len)
        """
        generated = start_token.clone()
        for _ in range(max_length - start_token.size(1)):
            logits = self.decode(generated)
            next_logits = logits[:, -1, :]
            if temperature!= 1.0:
                next_logits = next_logits / temperature
            if top_k > 0:
                topk_vals, topk_idx = torch.topk(next_logits, top_k)
                mask = torch.full_like(next_logits, float("-inf"))
                mask.scatter_(1, topk_idx, 0)
                next_logits = next_logits + mask
            if top_p > 0.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_logits[sorted_indices_to_remove] = float('-inf')
                next_logits.scatter_(1, sorted_indices, sorted_logits)
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
        return generated

__all__ = [
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
    "QTransformerTorchGen256",
]
