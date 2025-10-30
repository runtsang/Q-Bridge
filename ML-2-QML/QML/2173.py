import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from typing import Optional

class HybridTextClassifier(nn.Module):
    """
    Quantum‑augmented transformer‑based text classifier.
    When ``use_quantum=True`` the attention heads are transformed
    by a small PennyLane variational circuit.  All other components
    (feed‑forward, positional encoding, classification head) stay
    identical to the classical version, enabling a single API to
    evaluate both regimes.
    """
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int,
                 dropout: float = 0.1,
                 use_quantum: bool = False,
                 n_wires: int = 4,
                 q_device: str | None = None,
                 *_,  # ignore unused kwargs
                 ):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.blocks = nn.ModuleList([
            TransformerBlockQuantum(embed_dim,
                                    num_heads,
                                    ffn_dim,
                                    dropout,
                                    use_quantum,
                                    n_wires,
                                    q_device)
            for _ in range(num_blocks)
        ])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim,
                                    num_classes if num_classes > 2 else 1)

    def forward(self,
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.token_embed(x)
        x = self.pos_encoder(x)
        for block in self.blocks:
            x = block(x)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)

class PositionalEncoder(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float)
                             * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class QuantumAttention(nn.Module):
    """
    Multi‑head attention where each head’s projection is passed through
    a small variational circuit implemented with PennyLane.  The circuit
    encodes the input vector as rotations on each wire, applies two
    layers of trainable RX/RY gates, entangles with CNOTs, and finally
    measures the expectation of PauliZ on each wire.
    """
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float = 0.1,
                 n_wires: int = 4,
                 q_device: str | None = None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.n_wires = n_wires
        if n_wires > self.d_k:
            raise ValueError("n_wires must not exceed d_k")
        self.qdevice = qml.device("default.qubit",
                                  wires=self.n_wires,
                                  shots=1)
        # Linear projections
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        # Trainable parameters for the second layer
        self.params = nn.Parameter(torch.randn(self.n_wires))
        # Base circuit that encodes input angles
        def circuit(inputs, params):
            for i, angle in enumerate(inputs):
                qml.RX(angle, wires=i)
            for i, p in enumerate(params):
                qml.RX(p, wires=i)
                qml.RY(p, wires=i)
            # Entangling layer
            for i in range(self.n_wires - 1):
                qml.CNOT(wires=[i, i + 1])
            qml.CNOT(wires=[self.n_wires - 1, 0])
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_wires)]
        self.qnode = qml.QNode(circuit,
                               self.qdevice,
                               interface="torch")

    def _apply(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        tensor shape: (batch, seq, heads, d_k)
        """
        batch, seq, heads, d_k = tensor.shape
        # Flatten batch, seq, heads
        flat = tensor.view(-1, d_k)
        # Pad or truncate to match n_wires
        if d_k > self.n_wires:
            flat = flat[:, :self.n_wires]
        elif d_k < self.n_wires:
            pad = torch.zeros(flat.size(0), self.n_wires - d_k,
                              device=flat.device)
            flat = torch.cat([flat, pad], dim=1)
        out = self.qnode(flat, self.params)
        out = out.view(batch, seq, heads, self.n_wires)
        # If we padded, truncate back
        if self.n_wires > d_k:
            out = out[..., :d_k]
        return out

    def forward(self,
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Classical linear projections
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)
        # Reshape
        q = q.view(x.size(0), x.size(1), self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(x.size(0), x.size(1), self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(x.size(0), x.size(1), self.num_heads, self.d_k).transpose(1, 2)
        # Quantum transform
        q = self._apply(q)
        k = self._apply(k)
        v = self._apply(v)
        # Scaled dot‑product
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, v)
        # Merge heads
        context = context.transpose(1, 2).contiguous().view(x.size(0), x.size(1), self.embed_dim)
        return context

class TransformerBlockQuantum(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 ffn_dim: int,
                 dropout: float = 0.1,
                 use_quantum: bool = False,
                 n_wires: int = 4,
                 q_device: str | None = None):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        if use_quantum:
            self.attn = QuantumAttention(embed_dim,
                                         num_heads,
                                         dropout,
                                         n_wires,
                                         q_device)
        else:
            self.attn = nn.MultiheadAttention(embed_dim,
                                              num_heads,
                                              dropout=dropout,
                                              batch_first=True)
            class _Wrap(nn.Module):
                def __init__(self, attn): super().__init__(); self.attn = attn
                def forward(self, x, mask=None):
                    out, _ = self.attn(x, x, x, key_padding_mask=mask)
                    return out
            self.attn = _Wrap(self.attn)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim)
        )

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))
