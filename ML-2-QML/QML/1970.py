import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class MultiHeadAttentionBase(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self._head_dim = embed_dim // num_heads

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        return x.view(batch, seq, self.num_heads, self._head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, heads, seq, head_dim = x.shape
        return x.transpose(1, 2).contiguous().view(batch, seq, heads * head_dim)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__(embed_dim, num_heads, dropout)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq, _ = x.shape
        k = self.k_proj(x)
        q = self.q_proj(x)
        v = self.v_proj(x)

        k = self._split_heads(k)
        q = self._split_heads(q)
        v = self._split_heads(v)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self._head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = self._merge_heads(out)
        return self.out_proj(out)

class MultiHeadAttentionQuantum(MultiHeadAttentionBase):
    class _QuantumHead(tq.QuantumModule):
        def __init__(self, n_wires: int = 8):
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            self.parameters = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            for gate in self.parameters:
                gate(q_device)
            for i in range(self.n_wires - 1):
                tqf.cnot(q_device, wires=[i, i + 1])
            tqf.cnot(q_device, wires=[self.n_wires - 1, 0])
            return self.measure(q_device)

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, q_device: Optional[tq.QuantumDevice] = None):
        super().__init__(embed_dim, num_heads, dropout)
        self.q_head = self._QuantumHead()
        self.q_device = q_device or tq.QuantumDevice(n_wires=self.q_head.n_wires)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def _apply_quantum(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        out = torch.zeros(batch, seq, self.embed_dim, device=x.device, dtype=x.dtype)
        for i in range(batch):
            qdev = self.q_device.copy(bsz=1, device=x.device)
            token = x[i].unsqueeze(0)
            qout = self.q_head(token, qdev)
            out[i] = qout.squeeze(0)
        return out

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        k = self._apply_quantum(x)
        q = self._apply_quantum(x)
        v = self._apply_quantum(x)

        k = self._split_heads(k)
        q = self._split_heads(q)
        v = self._split_heads(v)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self._head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = self._merge_heads(out)
        return self.out_proj(out)

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
        self.linear1 = nn.Linear(embed_dim, ffn_dim, bias=True)
        self.linear2 = nn.Linear(ffn_dim, embed_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class FeedForwardQuantum(FeedForwardBase):
    class _QuantumFF(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            self.parameters = nn.ModuleList(
                [tq.RY(has_params=True, trainable=True) for _ in range(n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            for gate in self.parameters:
                gate(q_device)
            return self.measure(q_device)

    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int, dropout: float = 0.1):
        super().__init__(embed_dim, ffn_dim, dropout)
        self.q_ff = self._QuantumFF(n_qubits)
        self.q_device = tq.QuantumDevice(n_wires=n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim, bias=True)
        self.linear2 = nn.Linear(ffn_dim, embed_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        out = torch.zeros(batch, seq, self.q_ff.n_wires, device=x.device, dtype=x.dtype)
        for i in range(batch):
            qdev = self.q_device.copy(bsz=1, device=x.device)
            token = x[i].unsqueeze(0)
            out[i] = self.q_ff(token, qdev).squeeze(0)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))

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
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        n_qubits_transformer: int,
        n_qubits_ffn: int,
        n_qlayers: int,
        q_device: Optional[tq.QuantumDevice] = None,
        dropout: float = 0.1,
    ):
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout, q_device)
        if n_qubits_ffn > 0:
            self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits_ffn, dropout)
        else:
            self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class TransformerBlockHybrid(TransformerBlockBase):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        mode: str = "classical",
        n_qubits_transformer: int = 0,
        n_qubits_ffn: int = 0,
        n_qlayers: int = 1,
        q_device: Optional[tq.QuantumDevice] = None,
        dropout: float = 0.1,
    ):
        super().__init__(embed_dim, num_heads, dropout)
        if mode == "classical":
            self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
            self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)
        elif mode == "quantum_attn":
            self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout, q_device)
            self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)
        elif mode == "quantum_ffn":
            self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
            self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits_ffn, dropout)
        elif mode == "full_quantum":
            self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout, q_device)
            self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits_ffn, dropout)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

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

class QuantumPositionalEncoder(tq.QuantumModule):
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.n_wires = int(math.ceil(math.log2(embed_dim)))
        self.register_buffer("pe", self._precompute_pe())

    def _precompute_pe(self) -> torch.Tensor:
        pe = torch.zeros(self.max_len, self.embed_dim)
        for pos in range(self.max_len):
            qdev = tq.QuantumDevice(n_wires=self.n_wires)
            for wire in range(self.n_wires):
                if (pos >> wire) & 1:
                    tqf.x(qdev, wires=[wire])
            for i in range(self.n_wires):
                tqf.h(qdev, wires=[i])
                for j in range(i):
                    angle = math.pi / (2 ** (i - j))
                    tqf.cphase(qdev, gates=[angle], wires=[j, i])
            z = tq.MeasureAll(tq.PauliZ)(qdev)
            pe[pos] = z.squeeze(0).float()
        return pe.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, : seq_len]

class TextClassifier(nn.Module):
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
        n_qlayers: int = 1,
        q_device: Optional[tq.QuantumDevice] = None,
        mode: str = "classical",
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        if n_qubits_transformer > 0:
            self.pos_embedding = QuantumPositionalEncoder(embed_dim)
        else:
            self.pos_embedding = PositionalEncoder(embed_dim)
        blocks = []
        for _ in range(num_blocks):
            if mode == "classical":
                blocks.append(TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout))
            elif mode == "hybrid":
                blocks.append(
                    TransformerBlockHybrid(
                        embed_dim,
                        num_heads,
                        ffn_dim,
                        mode="full_quantum" if n_qubits_transformer > 0 else "classical",
                        n_qubits_transformer=n_qubits_transformer,
                        n_qubits_ffn=n_qubits_ffn,
                        n_qlayers=n_qlayers,
                        q_device=q_device,
                        dropout=dropout,
                    )
                )
            elif mode == "full_quantum":
                blocks.append(
                    TransformerBlockQuantum(
                        embed_dim,
                        num_heads,
                        ffn_dim,
                        n_qubits_transformer,
                        n_qubits_ffn,
                        n_qlayers,
                        q_device=q_device,
                        dropout=dropout,
                    )
                )
            else:
                raise ValueError(f"Unsupported mode: {mode}")
        self.transformers = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_embedding(tokens)
        x = self.transformers(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)

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
    "TransformerBlockHybrid",
    "PositionalEncoder",
    "QuantumPositionalEncoder",
    "TextClassifier",
]
