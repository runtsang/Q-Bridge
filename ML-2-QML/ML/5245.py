import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

# ------------------------------------------------------------------
# 1. Multi‑head attention primitives
# ------------------------------------------------------------------
class MultiHeadAttentionBase(nn.Module):
    """Base class that validates dimensionality and stores common hyper‑parameters."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    """Standard PyTorch multi‑head attention."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, seq, _ = x.size()
        q = self.q_proj(x).view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.d_k**0.5
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch, seq, self.embed_dim)
        return self.out_proj(out)


class MultiHeadAttentionQuantum(MultiHeadAttentionBase):
    """Quantum‑augmented attention that replaces linear projections with a small variational circuit."""
    class _QLayer(tq.QuantumModule):
        """Per‑head quantum sub‑module generating a k‑dim vector."""
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            # Encode each input feature into a qubit via RX
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            return self.measure(qdev)

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.q_layer = self._QLayer(self.d_k)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def _apply_qhead(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
        """Apply the quantum layer to every token, producing a d_k‑dim vector."""
        batch, seq, _ = x.size()
        # Reshape to feed each token independently into the quantum circuit
        tokens = x.view(batch * seq, self.d_k)
        qdev = qdev.copy(bsz=tokens.size(0), device=tokens.device)
        out = self.q_layer(tokens, qdev)
        return out.view(batch, seq, self.d_k)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, seq, _ = x.size()
        qdev = tq.QuantumDevice(n_wires=self.d_k, bsz=batch * seq, device=x.device)
        q = self._apply_qhead(x, qdev).view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        k = self._apply_qhead(x, qdev).view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        v = self._apply_qhead(x, qdev).view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.d_k**0.5
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch, seq, self.embed_dim)
        return self.out_proj(out)

# ------------------------------------------------------------------
# 2. Feed‑forward primitives
# ------------------------------------------------------------------
class FeedForwardBase(nn.Module):
    """Base for classical or quantum feed‑forward layers."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class FeedForwardClassical(FeedForwardBase):
    """Standard two‑layer MLP."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.lin1 = nn.Linear(embed_dim, ffn_dim, bias=False)
        self.lin2 = nn.Linear(ffn_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.dropout(F.relu(self.lin1(x))))


class FeedForwardQuantum(FeedForwardBase):
    """Quantum feed‑forward inspired by the QFCModel quantum layer."""
    class _QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            self.params = nn.ModuleList([tq.RY(has_params=True, trainable=True) for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            return self.measure(qdev)

    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.q_layer = self._QLayer(n_qubits)
        self.q_device = tq.QuantumDevice(n_wires=n_qubits)
        self.lin1 = nn.Linear(n_qubits, ffn_dim, bias=False)
        self.lin2 = nn.Linear(ffn_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.size()
        qdev = self.q_device.copy(bsz=batch * seq, device=x.device)
        out = self.q_layer(x.view(batch * seq, -1), qdev)
        out = out.view(batch, seq, -1)
        out = self.lin1(self.dropout(out))
        return self.lin2(F.relu(out))

# ------------------------------------------------------------------
# 3. Transformer block primitives
# ------------------------------------------------------------------
class TransformerBlockBase(nn.Module):
    """Common layer normalisation and dropout."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class TransformerBlockClassical(TransformerBlockBase):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class TransformerBlockQuantum(TransformerBlockBase):
    """Hybrid block that can mix classical and quantum sub‑modules."""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        n_qubits_attn: int | None = None,
        n_qubits_ffn: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout) if n_qubits_attn else MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits_ffn, dropout) if n_qubits_ffn else FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

# ------------------------------------------------------------------
# 4. Positional encoding
# ------------------------------------------------------------------
class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding compatible with both classical and quantum pipelines."""
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

# ------------------------------------------------------------------
# 5. Quantum‑convolutional preprocessing (optional)
# ------------------------------------------------------------------
class QuantumConvPreprocessor(nn.Module):
    """Thin wrapper around the Conv seed's quantum convolution filter."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        # Reuse the ConvFilter class from Conv seed
        from Conv import Conv  # local import to keep the module self‑contained
        self.conv_filter = Conv()  # returns a ConvFilter instance

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect a 4‑D tensor (batch, 1, H, W).  We flatten spatial dims into tokens.
        batch, _, H, W = x.size()
        tokens = []
        for i in range(0, H - self.conv_filter.kernel_size + 1, self.conv_filter.kernel_size):
            for j in range(0, W - self.conv_filter.kernel_size + 1, self.conv_filter.kernel_size):
                patch = x[:, :, i:i + self.conv_filter.kernel_size, j:j + self.conv_filter.kernel_size]
                # Convert patch to numpy for the quantum filter
                patch_np = patch.cpu().numpy()
                logits = self.conv_filter.run(patch_np)
                tokens.append(torch.tensor(logits, device=x.device, dtype=torch.float32))
        if not tokens:
            return torch.zeros(batch, 0, dtype=torch.float32, device=x.device)
        return torch.stack(tokens, dim=1)  # shape: (batch, seq, 1)

# ------------------------------------------------------------------
# 6. Hybrid text classifier
# ------------------------------------------------------------------
class HybridTextClassifier(nn.Module):
    """
    A transformer‑based classifier that can be instantiated in a fully classical
    mode or a hybrid quantum‑augmented mode.  The constructor accepts flags that
    enable quantum attention and/or quantum feed‑forward sub‑modules.  An optional
    quantum convolutional preprocessor turns image data into token embeddings.
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
        use_quantum: bool = False,
        n_qubits_attn: int | None = None,
        n_qubits_ffn: int | None = None,
        use_qconv: bool = False,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.use_qconv = use_qconv
        if use_qconv:
            self.preprocessor = QuantumConvPreprocessor()
        else:
            self.preprocessor = None

        blocks = []
        for _ in range(num_blocks):
            if use_quantum:
                blocks.append(
                    TransformerBlockQuantum(
                        embed_dim,
                        num_heads,
                        ffn_dim,
                        n_qubits_attn=n_qubits_attn,
                        n_qubits_ffn=n_qubits_ffn,
                        dropout=dropout,
                    )
                )
            else:
                blocks.append(
                    TransformerBlockClassical(
                        embed_dim, num_heads, ffn_dim, dropout
                    )
                )
        self.transformer = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass accepts either token indices (long) or raw image tensors.
        If use_qconv is True, the input is treated as image data and processed
        through a quantum convolutional preprocessor before tokenisation.
        """
        if self.use_qconv:
            # x shape: (batch, 1, H, W)
            tokens = self.preprocessor(x)
            # tokens are already embeddings; no token_embedding step
            x = tokens
        else:
            # x shape: (batch, seq_len)
            tokens = self.token_embedding(x)
            x = tokens

        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # global average pooling
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
    "PositionalEncoder",
    "QuantumConvPreprocessor",
    "HybridTextClassifier",
]
