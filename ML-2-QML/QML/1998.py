from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

# --------------------------------------------------------------------------- #
# 0.  Quantum‑aware attention
# --------------------------------------------------------------------------- #
class MultiHeadAttentionBase(nn.Module):
    """Shared logic for attention layers."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, bias: bool = False):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError(f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.bias = bias

    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

    def attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                  mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        return torch.matmul(scores, value), scores

    def downstream(self, query: torch.Tensor, key: torch.Tensor,
                   value: torch.Tensor, batch_size: int,
                   mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        q = self.separate_heads(query)
        k = self.separate_heads(key)
        v = self.separate_heads(value)
        out, _ = self.attention(q, k, v, mask)
        return out.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)

class MultiHeadAttentionQuantum(MultiHeadAttentionBase):
    """Multi‑head attention that applies a small variational circuit to each head."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int = 8):
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            # trainable rotation gates for each wire
            self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True)
                                         for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            # encode classical data as rotations
            self.encoder(q_device, x)
            for gate in self.params:
                gate(q_device)
            # build a simple entangling layer
            for i in range(self.n_wires - 1):
                tqf.cnot(q_device, wires=[i, i + 1])
            tqf.cnot(q_device, wires=[self.n_wires - 1, 0])
            return self.measure(q_device)

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1,
                 bias: bool = False, n_wires: int = 8):
        super().__init__(embed_dim, num_heads, dropout, bias)
        self.q_layer = self.QLayer(n_wires)
        self.combine = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _apply_quantum(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the quantum circuit to each token and head."""
        batch_size = x.size(0)
        seq_len = x.size(1)
        projections = []
        for token in torch.unbind(x, dim=1):
            # token: (batch, embed_dim)
            token = token.view(batch_size, self.num_heads, -1)
            head_outputs = []
            for head in torch.unbind(token, dim=1):
                qdev = tq.QuantumDevice(n_wires=self.q_layer.n_wires,
                                        bsz=head.size(0),
                                        device=head.device)
                head_outputs.append(self.q_layer(head, qdev))
            projections.append(torch.stack(head_outputs, dim=1))
        return torch.stack(projections, dim=1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        k = self._apply_quantum(x)
        q = self._apply_quantum(x)
        v = self._apply_quantum(x)
        out = self.downstream(q, k, v, x.size(0), mask)
        return self.combine(out)

# --------------------------------------------------------------------------- #
# 1.  Feed‑forward quantum
# --------------------------------------------------------------------------- #
class FeedForwardBase(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

class FeedForwardQuantum(FeedForwardBase):
    """Feed‑forward realised by a small variational quantum circuit."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int = 8):
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            self.params = nn.ModuleList([tq.RY(has_params=True, trainable=True)
                                         for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            for gate in self.params:
                gate(q_device)
            return self.measure(q_device)

    def __init__(self, embed_dim: int, ffn_dim: int, n_wires: int = 8, dropout: float = 0.1):
        super().__init__(embed_dim, ffn_dim, dropout)
        self.q_layer = self.QLayer(n_wires)
        self.linear1 = nn.Linear(n_wires, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.size()
        out = []
        for token in torch.unbind(x, dim=1):
            qdev = tq.QuantumDevice(n_wires=self.q_layer.n_wires,
                                    bsz=token.size(0),
                                    device=token.device)
            out.append(self.q_layer(token, qdev))
        out = torch.stack(out, dim=1)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))

class FeedForwardClassical(FeedForwardBase):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

# --------------------------------------------------------------------------- #
# 2.  Transformer block
# --------------------------------------------------------------------------- #
class TransformerBlockBase(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

class TransformerBlockQuantum(TransformerBlockBase):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 n_qheads: int = 8, n_ffn_wires: int = 8,
                 dropout: float = 0.1):
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads,
                                              dropout, n_wires=n_qheads)
        self.ffn = FeedForwardQuantum(embed_dim, ffn_dim,
                                      n_wires=n_ffn_wires, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class TransformerBlockClassical(TransformerBlockBase):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 dropout: float = 0.1):
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

# --------------------------------------------------------------------------- #
# 3.  Positional encoding
# --------------------------------------------------------------------------- #
class PositionalEncoder(nn.Module):
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
# 4.  Full model
# --------------------------------------------------------------------------- #
class QTransformerTorchGen(nn.Module):
    """Hybrid transformer that can swap classical and quantum sub‑modules."""
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        use_quantum: bool = False,
        n_qheads: int = 8,
        n_ffn_wires: int = 8,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            if use_quantum:
                self.blocks.append(TransformerBlockQuantum(embed_dim, num_heads,
                                                          ffn_dim,
                                                          n_qheads, n_ffn_wires,
                                                          dropout))
            else:
                self.blocks.append(TransformerBlockClassical(embed_dim, num_heads,
                                                            ffn_dim, dropout))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(x)
        x = self.pos_encoder(x)
        for block in self.blocks:
            x = block(x)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)

    # --------------------------------------------------------------------- #
    # 5.  Training step with quantum‑aware loss
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
        self.train()
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = self(inputs)
        loss = loss_fn(outputs.squeeze(-1) if outputs.ndim == 2 else outputs, targets)

        if fidelity_weight > 0.0:
            probs = torch.softmax(outputs, dim=-1).clamp(min=1e-7)
            target_onehot = torch.nn.functional.one_hot(targets,
                                                        num_classes=outputs.shape[-1]).float()
            fidelity = torch.nn.functional.cosine_similarity(probs, target_onehot, dim=-1).mean()
            loss = loss + fidelity_weight * (1.0 - fidelity)

        loss.backward()
        if log_gradients:
            for name, param in self.named_parameters():
                if param.grad is not None:
                    torch.nn.utils.clip_grad_norm_(param, max_norm=1.0)
        optimizer.step()
        return {"loss": loss.item(),
                "fidelity": fidelity.item() if fidelity_weight > 0 else None}

    def to_device(self, device: torch.device) -> None:
        self.to(device)

__all__ = ["QTransformerTorchGen"]
