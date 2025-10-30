import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple, Optional

# --------------------------------------------------------------------------- #
#  Classical fraud‑detection backbone – a linear stack that emulates the
#  photonic circuit from the original seed.  The architecture is kept
#  identical to the seed but is now wrapped in a reusable nn.Module.
# --------------------------------------------------------------------------- #
@dataclass
class FraudLayerParameters:
    """Parameters describing a fully‑connected layer that mimics the
    photonic layer.  The dataclass is identical to the seed but the
    methods that build the network are now part of the module."""
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
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
            return out * self.scale + self.shift

    return Layer()

class FraudDetector(nn.Module):
    """Linear stack that mirrors the photonic fraud‑detection circuit."""
    def __init__(self,
                 input_params: FraudLayerParameters,
                 hidden_params: Iterable[FraudLayerParameters],
                 final_dim: int = 1) -> None:
        super().__init__()
        modules = [_layer_from_params(input_params, clip=False)]
        modules.extend(_layer_from_params(p, clip=True) for p in hidden_params)
        modules.append(nn.Linear(2, final_dim))
        self.model = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

# --------------------------------------------------------------------------- #
#  Quantum‑enhanced transformer components (based on QTransformerTorch)
# --------------------------------------------------------------------------- #
import torchquantum as tq
import torchquantum.functional as tqf

class MultiHeadAttentionQuantum(nn.Module):
    """Quantum‑enhanced attention that maps each head through a variational circuit."""
    class QLayer(tq.QuantumModule):
        def __init__(self) -> None:
            super().__init__()
            self.n_wires = 8
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(self.n_wires)]
            )
            self.parameters = nn.ModuleList([tq.RX(has_params=True, trainable=True)
                                             for _ in range(self.n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            for wire, gate in enumerate(self.parameters):
                gate(q_device, wires=wire)
            for wire in range(self.n_wires - 1):
                tqf.cnot(q_device, wires=[wire, wire + 1])
            tqf.cnot(q_device, wires=[self.n_wires - 1, 0])
            return self.measure(q_device)

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float = 0.1,
                 q_device: Optional[tq.QuantumDevice] = None) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.n_wires = 8
        self.q_layer = self.QLayer()
        self.q_device = q_device or tq.QuantumDevice(n_wires=self.n_wires)
        self.combine = nn.Linear(embed_dim, embed_dim)

    def _apply_quantum_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, embed = x.shape
        d_k = embed // self.n_wires
        heads = x.view(batch, seq, self.n_wires, d_k).transpose(1, 2)  # (batch, n_heads, seq, d_k)
        out = []
        for h in heads.unbind(dim=1):
            qdev = self.q_device.copy(bsz=h.shape[0], device=h.device)
            out.append(self.q_layer(h, qdev))
        return torch.stack(out, dim=1)  # (batch, n_heads, seq, d_k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        heads = self._apply_quantum_heads(x)
        q, k, v = heads.chunk(3, dim=-1)  # split into Q, K, V
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.embed_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        return self.combine(out.transpose(1, 2).contiguous().view(x.size(0), x.size(1), -1))

class FeedForwardQuantum(nn.Module):
    """Quantum‑feed‑forward that uses a small variational circuit."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_qubits: int) -> None:
            super().__init__()
            self.n_qubits = n_qubits
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
            )
            self.parameters = nn.ModuleList([tq.RY(has_params=True, trainable=True)
                                             for _ in range(n_qubits)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            for wire, gate in enumerate(self.parameters):
                gate(q_device, wires=wire)
            return self.measure(q_device)

    def __init__(self,
                 embed_dim: int,
                 ffn_dim: int,
                 n_qubits: int,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.q_layer = self.QLayer(n_qubits)
        self.q_device = tq.QuantumDevice(n_wires=n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = []
        for token in x.unbind(dim=1):
            qdev = self.q_device.copy(bsz=token.size(0), device=token.device)
            out.append(self.q_layer(token, qdev))
        out = torch.stack(out, dim=1)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))

class TransformerBlockHybrid(nn.Module):
    """Hybrid block that optionally injects a quantum sub‑module."""
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 ffn_dim: int,
                 use_quantum: bool = False,
                 n_qubits: int = 0,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.attn = (MultiHeadAttentionQuantum(embed_dim, num_heads, dropout)
                     if use_quantum else nn.MultiheadAttention(embed_dim, num_heads,
                                                               dropout=dropout, batch_first=True))
        self.ffn = (FeedForwardQuantum(embed_dim, ffn_dim, n_qubits, dropout)
                    if use_quantum else nn.Sequential(
                        nn.Linear(embed_dim, ffn_dim), nn.ReLU(), nn.Linear(ffn_dim, embed_dim)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(self.attn, nn.MultiheadAttention):
            attn_out, _ = self.attn(x, x, x)
        else:
            attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class FraudTransformerHybrid(nn.Module):
    """Top‑level model that stitches together the classical fraud detector
    and a stack of transformer blocks.  The transformer can be pure
    classical or quantum‑enhanced, and the whole stack can be trained
    end‑to‑end or frozen for modular experiments."""
    def __init__(self,
                 fraud_params: FraudLayerParameters,
                 hidden_params: Iterable[FraudLayerParameters],
                 transformer_cfg: dict,
                 n_qubits: int = 0) -> None:
        super().__init__()
        self.fraud_detector = FraudDetector(fraud_params, hidden_params)
        self.transformer = nn.Sequential(
            *[TransformerBlockHybrid(**transformer_cfg) for _ in range(transformer_cfg["num_blocks"])]
        )
        self.classifier = nn.Linear(2, transformer_cfg.get("num_classes", 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fraud_out = self.fraud_detector(x)
        seq = fraud_out.unsqueeze(1)  # (batch, seq=1, dim=2)
        out = self.transformer(seq)
        out = out.mean(dim=1)
        return self.classifier(out)
