"""Hybrid transformer classifier with quantum attention, feed‑forward, and kernel support."""

from __future__ import annotations

import math
from typing import Iterable, Tuple, Optional

import torch
from torch import nn
from torch.nn import functional as F
import torchquantum as tq
from torchquantum.functional import func_name_dict, op_name_dict, cnot


# --------------------------------------------------------------------------- #
# 1. Autoencoder utilities (classical)
# --------------------------------------------------------------------------- #
class AutoencoderConfig:
    """Configuration for the fully‑connected autoencoder."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: Tuple[int, int] = (128, 64),
        dropout: float = 0.1,
    ) -> None:
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout


class AutoencoderNet(nn.Module):
    """Simple MLP autoencoder."""

    def __init__(self, cfg: AutoencoderConfig) -> None:
        super().__init__()
        encoder_layers = []
        in_dim = cfg.input_dim
        for h in cfg.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, h))
            encoder_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                encoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        encoder_layers.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = cfg.latent_dim
        for h in reversed(cfg.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, h))
            decoder_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                decoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        decoder_layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(x))


def Autoencoder(cfg: AutoencoderConfig) -> AutoencoderNet:
    """Factory returning a configured autoencoder."""
    return AutoencoderNet(cfg)


# --------------------------------------------------------------------------- #
# 2. Fraud‑detection sub‑network (classical)
# --------------------------------------------------------------------------- #
class FraudLayerParameters:
    """Parameters for a fraud‑detection layer."""

    def __init__(
        self,
        bs_theta: float,
        bs_phi: float,
        phases: Tuple[float, float],
        squeeze_r: Tuple[float, float],
        squeeze_phi: Tuple[float, float],
        displacement_r: Tuple[float, float],
        displacement_phi: Tuple[float, float],
        kerr: Tuple[float, float],
    ) -> None:
        self.bs_theta = bs_theta
        self.bs_phi = bs_phi
        self.phases = phases
        self.squeeze_r = squeeze_r
        self.squeeze_phi = squeeze_phi
        self.displacement_r = displacement_r
        self.displacement_phi = displacement_phi
        self.kerr = kerr


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


def _layer_from_params(params: FraudLayerParameters, clip: bool) -> nn.Module:
    weight = torch.tensor(
        [
            [params.bs_theta, params.bs_phi],
            [params.squeeze_r[0], params.squeeze_r[1]],
        ],
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
    """Create a sequential PyTorch model mirroring the photonic fraud‑detection circuit."""
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(l, clip=True) for l in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


# --------------------------------------------------------------------------- #
# 3. Kernel utilities (quantum)
# --------------------------------------------------------------------------- #
class KernalAnsatz(tq.QuantumModule):
    """Quantum data‑encoding ansatz that applies a list of rotations."""

    def __init__(self, func_list: list[dict]) -> None:
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = x[:, info["input_idx"]] if op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]] if op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)


class Kernel(tq.QuantumModule):
    """Quantum kernel that evaluates the overlap of two encoded states."""

    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = KernalAnsatz(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])


def kernel_matrix(a: Iterable[torch.Tensor], b: Iterable[torch.Tensor]) -> torch.Tensor:
    """Compute a Gram matrix using the quantum kernel."""
    kernel = Kernel()
    return torch.stack([torch.stack([kernel(x, y) for y in b]) for x in a])


# --------------------------------------------------------------------------- #
# 4. Transformer primitives (quantum aware)
# --------------------------------------------------------------------------- #
class MultiHeadAttentionBase(nn.Module):
    """Base class for multi‑head attention."""

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
        batch_size = x.size(0)
        return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

    def attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        return torch.matmul(scores, value), scores

    def downstream(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        batch_size: int,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = self.separate_heads(query)
        k = self.separate_heads(key)
        v = self.separate_heads(value)
        out, self.attn_weights = self.attention(q, k, v, mask)
        return out.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)


class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    """Standard multi‑head attention implemented with PyTorch's built‑in module."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.combine_heads = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        k = self.k_linear(x)
        q = self.q_linear(x)
        v = self.v_linear(x)
        return self.combine_heads(self.downstream(q, k, v, x.size(0), mask))


class MultiHeadAttentionQuantum(MultiHeadAttentionBase):
    """Variational multi‑head attention that routes each head through a small circuit."""

    class QHead(tq.QuantumModule):
        """Quantum circuit for a single attention head."""

        def __init__(self, n_wires: int = 8) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [i], "func": "rx", "wires": [i]}
                    for i in range(n_wires)
                ]
            )
            self.parameters = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            for wire, gate in enumerate(self.parameters):
                gate(q_device, wires=wire)
            # entangling pattern
            for wire in range(self.n_wires - 1):
                cnot(q_device, wires=[wire, wire + 1])
            cnot(q_device, wires=[self.n_wires - 1, 0])
            return self.measure(q_device)

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, q_device: Optional[tq.QuantumDevice] = None) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.q_head = self.QHead()
        self.q_device = q_device
        self.combine_heads = nn.Linear(embed_dim, embed_dim, bias=False)

    def _apply_quantum_heads(self, x: torch.Tensor) -> torch.Tensor:
        projections = []
        for token in x.unbind(dim=1):
            token = token.view(token.size(0), self.num_heads, -1)
            head_outputs = []
            for head in token.unbind(dim=1):
                qdev = self.q_device or tq.QuantumDevice(
                    n_wires=self.q_head.n_wires, bsz=head.size(0), device=head.device
                )
                head_outputs.append(self.q_head(head, qdev))
            projections.append(torch.stack(head_outputs, dim=1))
        return torch.stack(projections, dim=1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        k = self._apply_quantum_heads(x)
        q = self._apply_quantum_heads(x)
        v = self._apply_quantum_heads(x)
        return self.combine_heads(self.downstream(q, k, v, x.size(0), mask))


class FeedForwardBase(nn.Module):
    """Base class for the position‑wise feed‑forward network."""

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class FeedForwardClassical(FeedForwardBase):
    """Two‑layer MLP with ReLU activation."""

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class FeedForwardQuantum(FeedForwardBase):
    """Variational feed‑forward implemented with a small quantum circuit."""

    class QLayer(tq.QuantumModule):
        def __init__(self, n_qubits: int = 8) -> None:
            super().__init__()
            self.n_wires = n_qubits
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [idx], "func": "rx", "wires": [idx]}
                    for idx in range(n_qubits)
                ]
            )
            self.parameters = nn.ModuleList(
                [tq.RY(has_params=True, trainable=True) for _ in range(n_qubits)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            for wire, gate in enumerate(self.parameters):
                gate(q_device, wires=wire)
            return self.measure(q_device)

    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int = 8, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.q_layer = self.QLayer(n_qubits)
        self.q_device = tq.QuantumDevice(n_wires=n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        outputs = []
        for token in x.unbind(dim=1):
            qdev = self.q_device.copy(bsz=token.size(0), device=token.device)
            outputs.append(self.q_layer(token, qdev))
        out = torch.stack(outputs, dim=1)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))


class TransformerBlockBase(nn.Module):
    """Base transformer block containing attention and feed‑forward."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class TransformerBlockClassical(TransformerBlockBase):
    """Standard transformer block."""

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class TransformerBlockQuantum(TransformerBlockBase):
    """Transformer block that can optionally use quantum attention and feed‑forward."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        use_quantum: bool = False,
        n_qubits_ffn: int = 8,
        q_device: Optional[tq.QuantumDevice] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        if use_quantum:
            self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout, q_device=q_device)
            self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits=n_qubits_ffn, dropout=dropout)
        else:
            self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
            self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return x + self.pe[:, : x.size(1)]


# --------------------------------------------------------------------------- #
# 6. Hybrid text classifier (quantum aware)
# --------------------------------------------------------------------------- #
class HybridTextClassifier(nn.Module):
    """
    Transformer‑based classifier that can use quantum attention, feed‑forward,
    and a quantum kernel for the final logits.  The API mirrors the classical
    version so that the same experiment scripts can be used with either backend.
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
        *,
        use_autoencoder: bool = False,
        autoencoder_cfg: Optional[AutoencoderConfig] = None,
        use_fraud_detection: bool = False,
        fraud_layers: Optional[Iterable[FraudLayerParameters]] = None,
        use_kernel: bool = False,
        kernel_gamma: float = 1.0,
        use_quantum: bool = False,
        n_qubits_ffn: int = 8,
        q_device: Optional[tq.QuantumDevice] = None,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)

        # Optional autoencoder
        if use_autoencoder:
            if autoencoder_cfg is None:
                raise ValueError("autoencoder_cfg must be provided when use_autoencoder=True")
            self.autoencoder = Autoencoder(autoencoder_cfg)
        else:
            self.autoencoder = None

        # Transformer stack
        self.transformers = nn.Sequential(
            *[
                TransformerBlockQuantum(
                    embed_dim,
                    num_heads,
                    ffn_dim,
                    use_quantum=use_quantum,
                    n_qubits_ffn=n_qubits_ffn,
                    q_device=q_device,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )

        # Optional fraud detection branch
        if use_fraud_detection:
            if fraud_layers is None:
                raise ValueError("fraud_layers must be provided when use_fraud_detection=True")
            self.fraud_detection = build_fraud_detection_program(
                FraudLayerParameters(*([0.0] * 8)),  # placeholder input params
                fraud_layers,
            )
        else:
            self.fraud_detection = None

        # Final classifier
        self.dropout = nn.Dropout(dropout)
        if use_kernel:
            self.prototypes = nn.Parameter(torch.randn(num_classes, embed_dim))
            self.kernel = Kernel()
        else:
            self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        tokens = self.token_embedding(x)
        x = self.pos_embedding(tokens)
        x = self.transformers(x)
        x = x.mean(dim=1)

        if self.autoencoder is not None:
            x = self.autoencoder.encode(x)

        if self.fraud_detection is not None:
            fraud_score = self.fraud_detection(x.unsqueeze(1))
            x = x + fraud_score.squeeze()

        x = self.dropout(x)

        if hasattr(self, "kernel"):
            # Kernel‑based logits: similarity to class prototypes
            logits = torch.stack([self.kernel(x, proto) for proto in self.prototypes]).t()
            return logits
        else:
            return self.classifier(x)


__all__ = [
    "AutoencoderConfig",
    "AutoencoderNet",
    "Autoencoder",
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "KernalAnsatz",
    "Kernel",
    "kernel_matrix",
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
    "HybridTextClassifier",
]
