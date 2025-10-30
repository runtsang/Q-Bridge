import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Iterable, Sequence, Callable, List
import torchquantum as tq
import torchquantum.functional as tqf

# --------------------------------------------------------------------------- #
# Positional Encoding
# --------------------------------------------------------------------------- #
class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

# --------------------------------------------------------------------------- #
# Multi‑Head Attention
# --------------------------------------------------------------------------- #
class MultiHeadAttentionBase(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.size(0)
        return x.view(batch, -1, self.num_heads, self.d_k).transpose(1, 2)

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                  mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        return torch.matmul(scores, v), scores

    def downstream(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                   batch: int, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        q = self.separate_heads(q)
        k = self.separate_heads(k)
        v = self.separate_heads(v)
        out, _ = self.attention(q, k, v, mask)
        return out.transpose(1, 2).contiguous().view(batch, -1, self.embed_dim)

class MultiHeadAttentionQuantum(MultiHeadAttentionBase):
    """Quantum‑enhanced attention using a simple variational layer."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int = 8) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(qdev, x)
            for gate in self.params:
                gate(qdev)
            return self.measure(qdev)

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1,
                 q_device: Optional[tq.QuantumDevice] = None) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.q_layer = self.QLayer()
        self.q_device = q_device
        self.combine = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, _, _ = x.size()
        # Apply the same quantum projection to Q, K, V
        q = k = v = self._quantum_proj(x)
        out = self.downstream(q, k, v, batch, mask)
        return self.combine(out)

    def _quantum_proj(self, x: torch.Tensor) -> torch.Tensor:
        projections = []
        for token in x.unbind(dim=1):  # each token
            token = token.view(token.size(0), self.num_heads, -1)
            head_outs = []
            for head in token.unbind(dim=1):
                qdev = self.q_device or tq.QuantumDevice(self.q_layer.n_wires, bsz=head.size(0), device=head.device)
                head_outs.append(self.q_layer(head, qdev))
            projections.append(torch.stack(head_outs, dim=1))
        return torch.stack(projections, dim=1)

# --------------------------------------------------------------------------- #
# Feed‑Forward
# --------------------------------------------------------------------------- #
class FeedForwardBase(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

class FeedForwardQuantum(FeedForwardBase):
    """Quantum feed‑forward using a variational layer."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_qubits: int) -> None:
            super().__init__()
            self.n_qubits = n_qubits
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(n_qubits)]
            )
            self.params = nn.ModuleList([tq.RY(has_params=True, trainable=True) for _ in range(n_qubits)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(qdev, x)
            for gate in self.params:
                gate(qdev)
            return self.measure(qdev)

    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.q_layer = self.QLayer(n_qubits)
        self.q_device = tq.QuantumDevice(n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = []
        for token in x.unbind(dim=1):
            qdev = self.q_device.copy(bsz=token.size(0), device=token.device)
            outs.append(self.q_layer(token, qdev))
        out = torch.stack(outs, dim=1)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))

# --------------------------------------------------------------------------- #
# Transformer Block
# --------------------------------------------------------------------------- #
class TransformerBlockBase(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

class TransformerBlockQuantum(TransformerBlockBase):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 n_qubits_attn: int, n_qubits_ffn: int, dropout: float = 0.1,
                 q_device: Optional[tq.QuantumDevice] = None) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout, q_device=q_device)
        self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits_ffn, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

# --------------------------------------------------------------------------- #
# Quanvolution (Quantum)
# --------------------------------------------------------------------------- #
class QuanvolutionFilter(tq.QuantumModule):
    """Quantum kernel applied to 2×2 patches."""
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(self.n_wires)]
        )
        self.layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = torch.stack(
                    [x[:, r, c], x[:, r, c + 1], x[:, r + 1, c], x[:, r + 1, c + 1]],
                    dim=1,
                )
                self.encoder(qdev, patch)
                self.layer(qdev)
                patches.append(self.measure(qdev).view(bsz, 4))
        return torch.cat(patches, dim=1)

# --------------------------------------------------------------------------- #
# Auto‑Encoder (Quantum)
# --------------------------------------------------------------------------- #
class QuantumAutoencoder(nn.Module):
    """Simple quantum auto‑encoder using FeedForwardQuantum."""
    def __init__(self, embed_dim: int, latent_dim: int, n_qubits: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = FeedForwardQuantum(embed_dim, latent_dim, n_qubits, dropout)
        self.decoder = FeedForwardQuantum(latent_dim, embed_dim, n_qubits, dropout)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(x))

# --------------------------------------------------------------------------- #
# Fast Estimator (Quantum‑friendly)
# --------------------------------------------------------------------------- #
class FastBaseEstimator:
    """Evaluate a quantum‑aware model over batches of inputs and observables."""
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = torch.as_tensor(params, dtype=torch.float32).unsqueeze(0)
                outputs = self.model(inputs)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        scalar = float(val.mean().cpu())
                    else:
                        scalar = float(val)
                    row.append(scalar)
                results.append(row)
        return results

# --------------------------------------------------------------------------- #
# Hybrid Transformer (Quantum)
# --------------------------------------------------------------------------- #
class HybridTransformer(nn.Module):
    """
    Quantum‑enhanced transformer that can optionally prepend a quanvolution filter
    and append a quantum auto‑encoder.  The same ``HybridTransformer`` name is used
    in the classical module for API compatibility.
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
        use_quanvolution: bool = False,
        use_autoencoder: bool = False,
    ) -> None:
        super().__init__()
        self.use_quanvolution = use_quanvolution
        self.preprocessor = QuanvolutionFilter() if use_quanvolution else None

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlockQuantum(
                    embed_dim,
                    num_heads,
                    ffn_dim,
                    n_qubits_attn=embed_dim,  # simple mapping
                    n_qubits_ffn=embed_dim,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

        # Optional quantum auto‑encoder
        self.autoencoder: Optional[QuantumAutoencoder] = (
            QuantumAutoencoder(embed_dim, latent_dim=embed_dim // 2, n_qubits=embed_dim, dropout=dropout)
            if use_autoencoder
            else None
        )

    # --------------------------------------------------------------------- #
    # Forward
    # --------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.preprocessor is not None:
            x = self.preprocessor(x)
        tokens = self.token_embedding(x)
        x = self.pos_encoder(tokens)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)

    # --------------------------------------------------------------------- #
    # Auto‑encoding helpers
    # --------------------------------------------------------------------- #
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.autoencoder is None:
            raise RuntimeError("Quantum auto‑encoder not enabled")
        return self.autoencoder.encode(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if self.autoencoder is None:
            raise RuntimeError("Quantum auto‑encoder not enabled")
        return self.autoencoder.decode(z)

    # --------------------------------------------------------------------- #
    # Estimator
    # --------------------------------------------------------------------- #
    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        est = FastBaseEstimator(self)
        return est.evaluate(observables, parameter_sets)

__all__ = [
    "PositionalEncoder",
    "MultiHeadAttentionBase",
    "MultiHeadAttentionQuantum",
    "FeedForwardBase",
    "FeedForwardQuantum",
    "TransformerBlockBase",
    "TransformerBlockQuantum",
    "QuanvolutionFilter",
    "QuantumAutoencoder",
    "FastBaseEstimator",
    "HybridTransformer",
]
