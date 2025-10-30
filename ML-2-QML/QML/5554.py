"""Hybrid quantum regression model that mirrors the classical pipeline.

The quantum version replaces convolution, LSTM and transformer blocks with
variational circuits while preserving the same overall architecture.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
from torch.utils.data import Dataset


# --------------------------------------------------------------------------- #
# Data generation and dataset (identical to the classical version)
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


# --------------------------------------------------------------------------- #
# Quantum sub‑modules
# --------------------------------------------------------------------------- #
class QuantumConv(tq.QuantumModule):
    """Quantum 2‑D convolution implemented via a small variational circuit."""

    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.n_qubits = kernel_size ** 2

        # Simple parameterised circuit
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [i], "func": "rx", "wires": [i]}
                for i in range(self.n_qubits)
            ]
        )
        self.params = nn.ModuleList(
            [tq.RX(has_params=True, trainable=True) for _ in range(self.n_qubits)]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a single scalar per sample."""
        # Flatten to match qubit count
        data = x.view(-1, self.n_qubits)
        qdev = tq.QuantumDevice(n_wires=self.n_qubits, bsz=data.shape[0], device=data.device)
        self.encoder(qdev, data)
        for wire, gate in enumerate(self.params):
            gate(qdev, wires=wire)
        out = self.measure(qdev)
        return out.mean(dim=1)


class QuantumLSTM(tq.QuantumModule):
    """Quantum LSTM cell where each gate is a variational circuit."""

    class QGate(tq.QuantumModule):
        def __init__(self, n_wires: int) -> None:
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

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            for wire in range(self.n_wires - 1):
                tqf.cnot(qdev, wires=[wire, wire + 1])
            tqf.cnot(qdev, wires=[self.n_wires - 1, 0])
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Quantum gates for each LSTM gate
        self.forget_gate = self.QGate(n_qubits)
        self.input_gate = self.QGate(n_qubits)
        self.update_gate = self.QGate(n_qubits)
        self.output_gate = self.QGate(n_qubits)

        # Linear projections into qubit space
        self.forget_linear = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.input_linear = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.update_linear = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.output_linear = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(self, inputs: torch.Tensor, states: Optional[tuple[torch.Tensor, torch.Tensor]] = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_gate(self.forget_linear(combined)))
            i = torch.sigmoid(self.input_gate(self.input_linear(combined)))
            g = torch.tanh(self.update_gate(self.update_linear(combined)))
            o = torch.sigmoid(self.output_gate(self.output_linear(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        return torch.cat(outputs, dim=0), (hx, cx)

    def _init_states(self, inputs: torch.Tensor,
                     states: Optional[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))


class QuantumAttention(tq.QuantumModule):
    """Multi‑head attention where projections are processed by a quantum circuit."""

    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int) -> None:
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
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            return self.measure(qdev)

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float = 0.1,
                 n_wires_per_head: int = 4,
                 q_device: Optional[tq.QuantumDevice] = None) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

        self.q_layer = self.QLayer(n_wires_per_head)
        self.q_device = q_device or tq.QuantumDevice(n_wires=n_wires_per_head)
        self.combine = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = x.size(0)
        # Linear projections into qubit space
        proj = x.view(batch_size, -1, self.num_heads, self.d_k)
        proj = proj.permute(0, 2, 1, 3).contiguous()  # (B, H, L, d_k)
        # Process each head with a quantum circuit
        q_out = []
        for h in range(self.num_heads):
            head = proj[:, h, :, :].view(-1, self.d_k)
            qdev = self.q_device.copy(bsz=head.size(0), device=head.device)
            out = self.q_layer(head, qdev)
            q_out.append(out)
        q_out = torch.stack(q_out, dim=1)  # (B, H, L)
        # Simple dot‑product attention using the quantum‑processed keys
        scores = torch.matmul(q_out, q_out.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, q_out)
        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.embed_dim)
        return self.combine(out)


class QuantumFeedForward(tq.QuantumModule):
    """Feed‑forward network realised by a small variational circuit."""

    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [i], "func": "ry", "wires": [i]}
                    for i in range(n_wires)
                ]
            )
            self.params = nn.ModuleList(
                [tq.RY(has_params=True, trainable=True) for _ in range(n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            return self.measure(qdev)

    def __init__(self,
                 embed_dim: int,
                 ffn_dim: int,
                 n_wires: int,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_layer = self.QLayer(n_wires)
        self.q_device = tq.QuantumDevice(n_wires=n_wires)
        self.linear1 = nn.Linear(n_wires, ffn_dim)
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


class QuantumTransformerBlock(tq.QuantumModule):
    """Transformer block with quantum attention and feed‑forward."""

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 ffn_dim: int,
                 n_wires_attention: int,
                 n_wires_ffn: int,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = QuantumAttention(embed_dim, num_heads, dropout,
                                     n_wires_per_head=n_wires_attention)
        self.ffn = QuantumFeedForward(embed_dim, ffn_dim, n_wires_ffn, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class QuantumTransformerEncoder(tq.QuantumModule):
    """Stack of quantum transformer blocks."""

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 ffn_dim: int,
                 num_layers: int,
                 n_wires_attention: int,
                 n_wires_ffn: int,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            *[QuantumTransformerBlock(embed_dim, num_heads, ffn_dim,
                                      n_wires_attention, n_wires_ffn, dropout)
              for _ in range(num_layers)]
        )
        self.pos_encoder = PositionalEncoder(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pos_encoder(x)
        return self.blocks(x)


# --------------------------------------------------------------------------- #
# Hybrid quantum regression model
# --------------------------------------------------------------------------- #
class HybridRegressionModel(tq.QuantumModule):
    """Quantum‑enhanced regression model mirroring the classical architecture.

    Parameters
    ----------
    num_features : int
        Dimensionality of the raw input.
    conv_kernel : int
        Kernel size for the quantum convolution filter.
    lstm_hidden : int
        Hidden size of the quantum LSTM encoder.
    transformer_dim : int
        Embedding dimensionality for the quantum transformer.
    transformer_heads : int
        Number of attention heads.
    transformer_ffn : int
        Feed‑forward hidden size.
    transformer_layers : int
        Number of transformer blocks.
    dropout : float
        Dropout probability.
    use_lstm : bool
        Enable the quantum LSTM encoder.
    use_transformer : bool
        Enable the quantum transformer encoder.
    """

    def __init__(self,
                 num_features: int,
                 conv_kernel: int = 2,
                 lstm_hidden: int = 64,
                 transformer_dim: int = 64,
                 transformer_heads: int = 4,
                 transformer_ffn: int = 128,
                 transformer_layers: int = 2,
                 dropout: float = 0.1,
                 use_lstm: bool = True,
                 use_transformer: bool = True):
        super().__init__()
        self.conv = QuantumConv(kernel_size=conv_kernel)
        self.use_lstm = use_lstm
        self.use_transformer = use_transformer

        # Linear projection after convolution
        self.proj = nn.Linear(num_features, transformer_dim)

        if use_lstm:
            self.lstm = QuantumLSTM(num_features, lstm_hidden, n_qubits=8)
            lstm_out_dim = lstm_hidden
        else:
            lstm_out_dim = num_features

        if use_transformer:
            self.lstm_to_transformer = nn.Linear(lstm_out_dim, transformer_dim)
            self.transformer = QuantumTransformerEncoder(
                embed_dim=transformer_dim,
                num_heads=transformer_heads,
                ffn_dim=transformer_ffn,
                num_layers=transformer_layers,
                n_wires_attention=8,
                n_wires_ffn=8,
                dropout=dropout,
            )
            final_dim = transformer_dim
        else:
            final_dim = lstm_out_dim

        self.head = nn.Linear(final_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_out = self.conv(x)  # (batch,)
        conv_out = conv_out.unsqueeze(1)  # (batch, 1)
        embedded = self.proj(conv_out)

        if self.use_lstm:
            lstm_in = embedded.unsqueeze(1)
            lstm_out, _ = self.lstm(lstm_in)
            proj = self.lstm_to_transformer(lstm_out)
        else:
            proj = embedded.squeeze(1)

        if self.use_transformer:
            transformer_in = proj.unsqueeze(1)
            transformer_out = self.transformer(transformer_in)
            feat = transformer_out.squeeze(1)
        else:
            feat = proj

        return self.head(feat).squeeze(-1)


__all__ = [
    "generate_superposition_data",
    "RegressionDataset",
    "HybridRegressionModel",
]
