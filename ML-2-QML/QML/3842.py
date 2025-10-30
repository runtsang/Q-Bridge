"""
Quantum‑enhanced hybrid transformer.  Mirrors the classical
implementation but replaces selected sub‑modules with
variational quantum circuits.  The design follows the same
interface so that it can be swapped in as a drop‑in
replacement for the classical model.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf


# --------------------------------------------------------------------------- #
#  Quantum attention
# --------------------------------------------------------------------------- #

class MultiHeadAttentionQuantum(tq.QuantumModule):
    """Multi‑head attention implemented with a quantum encoder per head."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int = 8) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [idx], "func": "rx", "wires": [idx]}
                    for idx in range(n_wires)
                ]
            )
            self.parameters = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(qdev)
            for wire, gate in enumerate(self.parameters):
                gate(qdev, wires=wire)
            for wire in range(self.n_wires - 1):
                tqf.cnot(qdev, wires=[wire, wire + 1])
            tqf.cnot(qdev, wires=[self.n_wires - 1, 0])
            return self.measure(qdev)

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        q_device: Optional[tq.QuantumDevice] = None,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.q_layer = self.QLayer()
        self.q_device = q_device
        self.combine_heads = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        # Split heads
        heads = x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)  # B,H,L,D
        # Prepare quantum devices per head
        qdevs = [
            self.q_device or tq.QuantumDevice(n_wires=self.q_layer.n_wires, bsz=1, device=x.device)
            for _ in range(self.num_heads)
        ]
        # Encode each head into a quantum circuit and measure
        outputs = []
        for i, head in enumerate(heads.view(-1, self.d_k)):
            qdev = qdevs[i % self.num_heads]
            qdev.reset()
            self.q_layer(qdev)
            out = self.q_layer.measure(qdev)
            outputs.append(out)
        out = torch.stack(outputs).view(self.num_heads, batch_size, seq_len, self.d_k)
        out = out.transpose(0, 1)  # B,H,L,D
        out = out.reshape(batch_size, seq_len, self.embed_dim)
        return self.combine_heads(out)


# --------------------------------------------------------------------------- #
#  Quantum feed‑forward
# --------------------------------------------------------------------------- #

class FeedForwardQuantum(tq.QuantumModule):
    """Feed‑forward network realised by a variational quantum circuit."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_qubits: int) -> None:
            super().__init__()
            self.n_qubits = n_qubits
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

        def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(qdev)
            for wire, gate in enumerate(self.parameters):
                gate(qdev, wires=wire)
            return self.measure(qdev)

    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)
        self.q_layer = self.QLayer(n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for token in x.unbind(dim=1):
            qdev = tq.QuantumDevice(n_wires=self.q_layer.n_qubits, bsz=1, device=x.device)
            out = self.q_layer(qdev)
            outputs.append(out)
        out = torch.stack(outputs).reshape(x.shape[0], x.shape[1], -1)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))


# --------------------------------------------------------------------------- #
#  Transformer block and positional encoding
# --------------------------------------------------------------------------- #

class TransformerBlockQuantum(tq.QuantumModule):
    """Quantum‑enabled transformer block."""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        n_qubits_transformer: int,
        n_qubits_ffn: int,
        q_device: Optional[tq.QuantumDevice] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout, q_device)
        self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits_ffn, dropout)

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
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


# --------------------------------------------------------------------------- #
#  Quantum hybrid classifier / regressor
# --------------------------------------------------------------------------- #

class QuantumHybridTextClassifier(tq.QuantumModule):
    """
    Quantum‑enhanced transformer for classification or regression.
    Parameters are analogous to :class:`HybridTextClassifier`, but
    the transformer blocks are quantum‑enabled.
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
        n_qubits_transformer: int = 8,
        n_qubits_ffn: int = 8,
        q_device: Optional[tq.QuantumDevice] = None,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        blocks = [
            TransformerBlockQuantum(
                embed_dim,
                num_heads,
                ffn_dim,
                n_qubits_transformer,
                n_qubits_ffn,
                q_device,
                dropout=dropout,
            )
            for _ in range(num_blocks)
        ]
        self.transformer = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 1 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        tokens = self.token_embedding(x)
        x = self.pos_encoder(tokens)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)


# --------------------------------------------------------------------------- #
#  Quantum regression dataset and model
# --------------------------------------------------------------------------- #

def generate_superposition_data(num_wires: int, samples: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate quantum superposition states and labels."""
    omega_0 = torch.zeros(2 ** num_wires, dtype=torch.cdouble)
    omega_0[0] = 1.0
    omega_1 = torch.zeros(2 ** num_wires, dtype=torch.cdouble)
    omega_1[-1] = 1.0

    thetas = 2 * math.pi * torch.rand(samples)
    phis = 2 * math.pi * torch.rand(samples)
    states = torch.zeros((samples, 2 ** num_wires), dtype=torch.cdouble)
    for i in range(samples):
        states[i] = torch.cos(thetas[i]) * omega_0 + torch.exp(1j * phis[i]) * torch.sin(thetas[i]) * omega_1
    labels = torch.sin(2 * thetas) * torch.cos(phis)
    return states, labels


class RegressionDataset(torch.utils.data.Dataset):
    """Quantum dataset for regression tasks."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:  # type: ignore[override]
        return {"states": self.states[idx], "target": self.labels[idx]}


class QModel(tq.QuantumModule):
    """Quantum regression model mirroring the classical counterpart."""
    class QLayer(tq.QuantumModule):
        def __init__(self, num_wires: int):
            super().__init__()
            self.n_wires = num_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.random_layer(qdev)
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)

    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self.QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)


__all__ = [
    "MultiHeadAttentionQuantum",
    "FeedForwardQuantum",
    "TransformerBlockQuantum",
    "PositionalEncoder",
    "QuantumHybridTextClassifier",
    "RegressionDataset",
    "QModel",
]
