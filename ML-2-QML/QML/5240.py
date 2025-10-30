from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from dataclasses import dataclass
from typing import Iterable, Tuple

# --------------------------------------------------------------------------- #
# Fraud‑style parameterised layers – kept for consistency across variants
# --------------------------------------------------------------------------- #
@dataclass
class FraudLayerParameters:
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
    # Build a tiny quantum layer that mirrors the classical fraud‑layer
    class QLayer(tq.QuantumModule):
        def __init__(self):
            super().__init__()
            self.n_wires = 2
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [0], "func": "rx", "wires": [0]},
                    {"input_idx": [1], "func": "rx", "wires": [1]},
                ]
            )
            self.params = nn.ModuleList(
                [
                    tq.RX(has_params=True, trainable=True),
                    tq.RY(has_params=True, trainable=True),
                ]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(qdev, x)
            for gate, wire in zip(self.params, range(self.n_wires)):
                gate(qdev, wires=wire)
            return self.measure(qdev)

    return QLayer()


# --------------------------------------------------------------------------- #
# Quantum transformer primitives
# --------------------------------------------------------------------------- #
class MultiHeadAttentionQuantum(nn.Module):
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [i], "func": "rx", "wires": [i]}
                    for i in range(n_wires)
                ]
            )
            self.params = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(qdev, x)
            for gate, wire in zip(self.params, range(self.n_wires)):
                gate(qdev, wires=wire)
            return self.measure(qdev)

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.q_layer = self.QLayer(self.num_heads)
        self.combine_heads = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        # split into heads
        q = x.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        k = x.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        v = x.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)

        # Quantum projection for each head
        proj = []
        for head_idx in range(self.num_heads):
            head = q[:, head_idx]
            qdev = tq.QuantumDevice(n_wires=self.q_layer.n_wires, bsz=head.size(0), device=head.device)
            proj.append(self.q_layer(head, qdev))
        proj = torch.stack(proj, dim=1)  # (batch, heads, seq, n_wires)
        # Treat measurement as a linear projection
        proj = proj.mean(-1)  # (batch, heads, seq)
        # Re‑assemble
        proj = proj.view(batch_size, seq_len, self.embed_dim)
        return self.combine_heads(proj)


class FeedForwardQuantum(nn.Module):
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            self.params = nn.ModuleList(
                [tq.RY(has_params=True, trainable=True) for _ in range(n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(qdev, x)
            for gate, wire in zip(self.params, range(self.n_wires)):
                gate(qdev, wires=wire)
            return self.measure(qdev)

    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int):
        super().__init__()
        self.q_layer = self.QLayer(n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        out = []
        for seq_idx in range(seq_len):
            token = x[:, seq_idx]
            qdev = tq.QuantumDevice(n_wires=self.q_layer.n_wires, bsz=token.size(0), device=token.device)
            out.append(self.q_layer(token, qdev))
        out = torch.stack(out, dim=1)  # (batch, seq, n_qubits)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))


class TransformerBlockQuantum(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, n_qubits: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads)
        self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(self.norm1(x))
        x = x + self.dropout(attn_out)
        ffn_out = self.ffn(self.norm2(x))
        return x + self.dropout(ffn_out)


class PositionalEncoder(nn.Module):
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
# Quantum sampler – variational circuit returning probability amplitudes
# --------------------------------------------------------------------------- #
class QuantumSampler(tq.QuantumModule):
    def __init__(self, n_wires: int = 2):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
            ]
        )
        self.parameters = nn.ModuleList(
            [tq.RY(has_params=True, trainable=True) for _ in range(n_wires)]
        )
        self.cnot = tq.CNOT
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, inputs: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
        self.encoder(qdev, inputs)
        for gate, wire in zip(self.parameters, range(self.n_wires)):
            gate(qdev, wires=wire)
        self.cnot(qdev, wires=[0, 1])
        return self.measure(qdev)


# --------------------------------------------------------------------------- #
# Main hybrid sampler model – quantum variant
# --------------------------------------------------------------------------- #
class SamplerQNNGen124(nn.Module):
    """
    Quantum‑enhanced sampler that mirrors the classical SamplerQNNGen124 but
    replaces attention and feed‑forward sub‑modules with variational quantum
    circuits.  A dedicated QuantumSampler is also exposed for stand‑alone
    sampling experiments.
    """

    def __init__(
        self,
        embed_dim: int = 32,
        num_heads: int = 4,
        ffn_dim: int = 64,
        num_blocks: int = 2,
        n_qubits_transformer: int = 8,
        n_qubits_ffn: int = 8,
    ) -> None:
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(nn.Linear(16 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, embed_dim))
        self.norm = nn.BatchNorm1d(embed_dim)

        # Quantum transformer stack
        self.transformer = nn.Sequential(
            *[
                TransformerBlockQuantum(embed_dim, num_heads, ffn_dim, n_qubits_transformer)
                for _ in range(num_blocks)
            ]
        )
        self.classifier = nn.Linear(embed_dim, 2)

        # Stand‑alone quantum sampler
        self.quantum_sampler = QuantumSampler(n_wires=n_qubits_transformer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.feature_extractor(x)
        flattened = features.view(features.size(0), -1)
        embed = self.fc(flattened)
        embed = self.norm(embed)
        seq = embed.unsqueeze(1)  # (batch, 1, embed_dim)
        seq = self.transformer(seq)
        pooled = seq.mean(dim=1)
        logits = self.classifier(pooled)
        return F.softmax(logits, dim=-1)

    def sample(self, inputs: torch.Tensor, batch_size: int = 32) -> torch.Tensor:
        """
        Execute the quantum sampler circuit on the provided inputs.
        `inputs` should have shape (batch_size, 2) and contain real‑valued angles.
        Returns the probability distribution over the computational basis.
        """
        qdev = tq.QuantumDevice(n_wires=self.quantum_sampler.n_wires, bsz=batch_size, device=inputs.device)
        probs = self.quantum_sampler(inputs, qdev)
        return probs

__all__ = [
    "FraudLayerParameters",
    "SamplerQNNGen124",
]
