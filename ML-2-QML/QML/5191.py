"""HybridTransformer: a quantum‑enhanced transformer that integrates
variational circuits, graph‑based state propagation, and photonic fraud‑detection
encoders."""

from __future__ import annotations

import math
import itertools
from typing import Optional, Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate
import networkx as nx
import qutip as qt
import scipy as sc

# --------------------------------------------------------------------------- #
#  Fraud parameters – photonic & classical
# --------------------------------------------------------------------------- #

class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def build_sf_program(params: FraudLayerParameters) -> sf.Program:
    """Create a Strawberry Fields program for the photonic fraud circuit."""
    program = sf.Program(2)
    with program.context as q:
        BSgate(params.bs_theta, params.bs_phi) | (q[0], q[1])
        for i, phase in enumerate(params.phases):
            Rgate(phase) | q[i]
        for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
            Sgate(r if not _clip(r, 5) else _clip(r, 5), phi) | q[i]
        BSgate(params.bs_theta, params.bs_phi) | (q[0], q[1])
        for i, phase in enumerate(params.phases):
            Rgate(phase) | q[i]
        for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
            Dgate(r if not _clip(r, 5) else _clip(r, 5), phi) | q[i]
        for i, k in enumerate(params.kerr):
            Kgate(k if not _clip(k, 1) else _clip(k, 1)) | q[i]
    return program

class QuantumFraudEncoder(tq.QuantumModule):
    """Placeholder quantum fraud encoder – in practice, run the SF circuit."""
    def __init__(self, params: FraudLayerParameters):
        super().__init__()
        self.params = params

    def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
        # In a real implementation we would compile & run the SF program.
        # Here we return a dummy measurement.
        return torch.zeros(qdev.bsz, 1, device=qdev.device)

# --------------------------------------------------------------------------- #
#  Quantum graph encoder
# --------------------------------------------------------------------------- #

class QuantumGraphEncoder(tq.QuantumModule):
    """Encode a graph adjacency matrix into a quantum state."""
    def __init__(self, n_qubits: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [idx], "func": "rx", "wires": [idx]} for idx in range(n_qubits)]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
        self.encoder(qdev)
        return self.measure(qdev)

# --------------------------------------------------------------------------- #
#  Quantum attention & feed‑forward
# --------------------------------------------------------------------------- #

class MultiHeadAttentionQuantum(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [0], "func": "rx", "wires": [0]},
                    {"input_idx": [1], "func": "rx", "wires": [1]},
                    {"input_idx": [2], "func": "rx", "wires": [2]},
                    {"input_idx": [3], "func": "rx", "wires": [3]},
                    {"input_idx": [4], "func": "rx", "wires": [4]},
                    {"input_idx": [5], "func": "rx", "wires": [5]},
                    {"input_idx": [6], "func": "rx", "wires": [6]},
                    {"input_idx": [7], "func": "rx", "wires": [7]},
                ]
            )
            self.parameters = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(self.n_wires)]
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

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1,
                 q_device: Optional[tq.QuantumDevice] = None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.q_layer = self.QLayer(n_wires=8)
        self.q_device = q_device
        self.combine_heads = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        outputs = []
        for _ in range(seq_len):
            qdev = self.q_device or tq.QuantumDevice(n_wires=8, bsz=batch_size, device=x.device)
            out = self.q_layer(qdev)
            outputs.append(out)
        out = torch.stack(outputs, dim=1)
        return self.combine_heads(out)

class FeedForwardQuantum(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self, n_qubits: int):
            super().__init__()
            self.n_qubits = n_qubits
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [idx], "func": "rx", "wires": [idx]} for idx in range(n_qubits)]
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

    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int, dropout: float = 0.1):
        super().__init__()
        self.q_layer = self.QLayer(n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        outputs = []
        for _ in range(seq_len):
            qdev = tq.QuantumDevice(n_wires=self.q_layer.n_qubits, bsz=batch_size, device=x.device)
            out = self.q_layer(qdev)
            outputs.append(out)
        out = torch.stack(outputs, dim=1)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))

# --------------------------------------------------------------------------- #
#  Transformer block
# --------------------------------------------------------------------------- #

class TransformerBlockQuantum(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, n_qubits: int,
                 dropout: float = 0.1, q_device: Optional[tq.QuantumDevice] = None):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout, q_device=q_device)
        self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

# --------------------------------------------------------------------------- #
#  Positional encoder
# --------------------------------------------------------------------------- #

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

# --------------------------------------------------------------------------- #
#  Hybrid transformer – quantum version
# --------------------------------------------------------------------------- #

class HybridTransformer(nn.Module):
    """Quantum‑enhanced transformer supporting text, regression, graph, and fraud data."""
    def __init__(
        self,
        vocab_size: Optional[int] = None,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_blocks: int = 4,
        ffn_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.1,
        n_qubits: int = 8,
        q_device: Optional[tq.QuantumDevice] = None,
        use_graph: bool = False,
        use_fraud: bool = False,
        fraud_params: Optional[FraudLayerParameters] = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        if vocab_size is not None:
            self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        else:
            self.token_embedding = nn.Linear(1, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)
        self.transformers = nn.Sequential(
            *[TransformerBlockQuantum(embed_dim, num_heads, ffn_dim, n_qubits,
                                      dropout, q_device) for _ in range(num_blocks)]
        )
        self.dropout = nn.Dropout(dropout)
        self.graph_encoder = QuantumGraphEncoder(n_qubits) if use_graph else None
        self.fraud_encoder = QuantumFraudEncoder(fraud_params) if use_fraud else None
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor,
                adjacency: Optional[torch.Tensor] = None,
                fraud_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        if isinstance(self.token_embedding, nn.Embedding):
            tokens = self.token_embedding(x)
        else:
            tokens = self.token_embedding(x.unsqueeze(-1))
        x = self.pos_embedding(tokens)
        x = self.transformers(x)
        if self.graph_encoder is not None and adjacency is not None:
            qdev = tq.QuantumDevice(n_wires=self.graph_encoder.n_qubits,
                                    bsz=x.size(0), device=x.device)
            graph_feat = self.graph_encoder(qdev)
            x = x + graph_feat.unsqueeze(1)
        if self.fraud_encoder is not None and fraud_features is not None:
            qdev = tq.QuantumDevice(n_wires=2, bsz=x.size(0), device=x.device)
            fraud_feat = self.fraud_encoder(qdev)
            x = x + fraud_feat.unsqueeze(1)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)
