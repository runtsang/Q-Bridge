import math
from dataclasses import dataclass
from typing import Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional quantum imports – will be ignored if torchquantum is not installed
try:
    import torch.quantum as tq
    import torchquantum.functional as tqf
except Exception:
    tq = None
    tqf = None


@dataclass
class FraudLayerParameters:
    """Parameters describing a single photonic layer."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


def _clip(value: float, bound: float) -> float:
    """Clip a scalar value to the range [-bound, bound]."""
    return max(-bound, min(bound, value))


def _layer_from_params(params: FraudLayerParameters, *, clip: bool = True) -> nn.Module:
    """Create a 2‑node linear layer that mimics the photonic operations."""
    weight = torch.tensor(
        [[params.bs_theta, params.bs_phi],
         [params.squeeze_r[0], params.squeeze_r[1]]],
        dtype=torch.float32,
    )
    bias = torch.tensor(params.phases, dtype=torch.float32)
    if clip:
        weight = weight.clamp(-5.0, 5.0)
        bias = bias.clamp(-5.0, 5.0)

    linear = nn.Linear(2, 2)
    with torch.no_grad():
        linear.weight.copy_(weight)
        linear.bias.copy_(bias)

    activation = nn.Tanh()
    scale = torch.tensor(params.displacement_r, dtype=torch.float32)
    shift = torch.tensor(params.displacement_phi, dtype=torch.float32)

    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = linear
            self.activation = activation
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            out = self.activation(self.linear(inputs))
            out = out * self.scale + self.shift
            return out

    return Layer()


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Build a sequential neural chain that mirrors the photonic stack."""
    modules: List[nn.Module] = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(l, clip=True) for l in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


# --------------------------------------------------------------------------- #
#  Transformer‑style hybrid with optional quantum sub‑modules
# --------------------------------------------------------------------------- #

class MultiHeadAttentionBase(nn.Module):
    """Shared logic for attention layers."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.attn_weights: Optional[torch.Tensor] = None

    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        bs = x.size(0)
        return x.view(bs, -1, self.num_heads, self.d_k).transpose(1, 2)

    def attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        return torch.matmul(scores, value), scores

    def downstream(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        batch_size: int,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self.separate_heads(q)
        k = self.separate_heads(k)
        v = self.separate_heads(v)
        out, self.attn_weights = self.attention(q, k, v, mask)
        return out.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)


class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    """Standard multi‑head attention."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.combine_heads = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        k = self.k_linear(x)
        q = self.q_linear(x)
        v = self.v_linear(x)
        return self.combine_heads(self.downstream(q, k, v, x.size(0), mask))


class MultiHeadAttentionQuantum(MultiHeadAttentionBase):
    """Quantum‑encoded attention: each head is a variational circuit."""
    if tq is None:
        raise RuntimeError("torchquantum is not available – quantum attention cannot be used.")

    class _QLayer(tq.QuantumModule):
        def __init__(self) -> None:
            super().__init__()
            self.n_wires = 8
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(self.n_wires)]
            )
            self.param_gates = nn.ModuleList(
                [tqf.RX(has_params=True, trainable=True) for _ in range(self.n_wires)]
            )
            self.measure = tq.MeasureAll(tqf.PauliZ)

        def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(qdev, x)
            for gate, wire in zip(self.param_gates, range(self.n_wires)):
                gate(qdev, wires=wire)
            return self.measure(qdev)

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float = 0.1,
                 q_device: tq.QuantumDevice | None = None) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.q_layer = self._QLayer()
        self.q_device = q_device
        self.combine_heads = nn.Linear(embed_dim, embed_dim)

    def _apply_quantum_heads(self, x: torch.Tensor) -> torch.Tensor:
        projections = []
        for token in x.unbind(dim=1):
            token = token.view(token.size(0), self.num_heads, -1)
            head_out = []
            for head in token.unbind(dim=1):
                qdev = self.q_device or tq.QuantumDevice(
                    n_wires=self.q_layer.n_wires,
                    bsz=token.size(0),
                    device=head.device,
                )
                head_out.append(self.q_layer(head, qdev))
            projections.append(torch.stack(head_out, dim=1))
        return torch.stack(projections, dim=1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        k = self._apply_quantum_heads(x)
        q = self._apply_quantum_heads(x)
        v = self._apply_quantum_heads(x)
        return self.combine_heads(self.downstream(q, k, v, x.size(0), mask))


class FeedForwardBase(nn.Module):
    """Base feed‑forward layer."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)


class FeedForwardClassical(FeedForwardBase):
    """Two‑layer MLP."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class FeedForwardQuantum(FeedForwardBase):
    """Variational quantum feed‑forward."""
    if tq is None:
        raise RuntimeError("torchquantum is not available – quantum feed‑forward cannot be used.")

    class _QLayer(tq.QuantumModule):
        def __init__(self, n_qubits: int) -> None:
            super().__init__()
            self.n_qubits = n_qubits
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
            )
            self.param_gates = nn.ModuleList(
                [tqf.RY(has_params=True, trainable=True) for _ in range(n_qubits)]
            )
            self.measure = tq.MeasureAll(tqf.PauliZ)

        def forward(self, input_vec: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(qdev, input_vec)
            for gate, wire in zip(self.param_gates, range(self.n_qubits)):
                gate(qdev, wires=wire)
            return self.measure(qdev)

    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        n_qubits: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.q_layer = self._QLayer(n_qubits)
        self.q_device = tq.QuantumDevice(n_wires=n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = []
        for token in x.unbind(dim=1):
            qdev = self.q_device.copy(bsz=token.size(0), device=token.device)
            out.append(self.q_layer(token, qdev))
        out = torch.stack(out, dim=1)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))


class TransformerBlockBase(nn.Module):
    """Base transformer block."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)


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
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        n_qubits_transformer: int,
        n_qubits_ffn: int,
        q_device: tq.QuantumDevice | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionQuantum(
            embed_dim,
            num_heads,
            dropout,
            q_device=q_device,
        )
        self.ffn = FeedForwardQuantum(
            embed_dim,
            ffn_dim,
            n_qubits_ffn,
            dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / embed_dim)
        )
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class FraudTransformerHybrid(nn.Module):
    """
    A hybrid model that first passes the input through a photonic fraud‑detection core
    and then processes the resulting scalar feature with a transformer encoder.
    Classical or quantum sub‑modules can be selected via the ``use_quantum`` flag.
    """
    def __init__(
        self,
        fraud_input_params: FraudLayerParameters,
        fraud_layer_params: Iterable[FraudLayerParameters],
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        use_quantum: bool = False,
        n_qubits_transformer: int = 0,
        n_qubits_ffn: int = 0,
        q_device: tq.QuantumDevice | None = None,
    ) -> None:
        super().__init__()
        # Photonic fraud‑detection core
        self.fraud_core = build_fraud_detection_program(
            fraud_input_params, fraud_layer_params
        )
        # Mapping from scalar output to transformer embedding
        self.fraud_to_embed = nn.Linear(1, embed_dim)

        # Build transformer layers
        self.transformer_layers = nn.ModuleList()
        for _ in range(num_blocks):
            if use_quantum:
                block = TransformerBlockQuantum(
                    embed_dim,
                    num_heads,
                    ffn_dim,
                    n_qubits_transformer,
                    n_qubits_ffn,
                    q_device=q_device,
                )
            else:
                block = TransformerBlockClassical(embed_dim, num_heads, ffn_dim)
            self.transformer_layers.append(block)

        self.pos_encoder = PositionalEncoder(embed_dim)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : Tensor of shape (batch, 2) – raw fraud features.
        """
        # Photonic core
        fraud_out = self.fraud_core(x)          # (batch, 1)
        # Embed into transformer space
        emb = self.fraud_to_embed(fraud_out)    # (batch, embed_dim)
        seq = emb.unsqueeze(1)                 # (batch, 1, embed_dim)

        # Apply transformer layers
        for layer in self.transformer_layers:
            seq = layer(seq)

        seq = self.pos_encoder(seq)
        seq = self.dropout(seq)
        # Global average pooling over sequence dimension
        pooled = seq.mean(dim=1)
        logits = self.classifier(pooled)
        return logits


__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
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
    "FraudTransformerHybrid",
]
