"""HybridTransformer – quantum‑enhanced implementation using torchquantum.

The QML module mirrors the API of the classical module but replaces selected
transformer sub‑components with quantum circuits.  It supports a hybrid mode
where classical blocks are used by default, and optional quantum blocks can
be activated via *n_qubits_* parameters.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import StatevectorEstimator as StatevectorEstimator


# --------------------------------------------------------------------------- #
#  Core components – quantum
# --------------------------------------------------------------------------- #
class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (identical to the classical version)."""

    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim)
        )
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return x + self.pe[:, : x.size(1)]


class MultiHeadAttentionQuantum(nn.Module):
    """Quantum‑enhanced multi‑head attention.

    Each head is a small quantum circuit that encodes the input token and
    applies parameterised RX gates followed by a short CNOT ladder.  The
    measurement yields a real‑valued vector that is then linearly combined.
    """

    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int = 8) -> None:
            super().__init__()
            self.n_wires = n_wires
            # Encode each input value onto a distinct qubit via RX
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

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            for wire, gate in enumerate(self.parameters):
                gate(q_device, wires=wire)
            # short entangling ladder
            for wire in range(self.n_wires - 1):
                tqf.cnot(q_device, wires=[wire, wire + 1])
            tqf.cnot(q_device, wires=[self.n_wires - 1, 0])
            return self.measure(q_device)

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        n_wires: int = 8,
        q_device: Optional[tq.QuantumDevice] = None,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.q_layer = self.QLayer(n_wires)
        self.q_device = q_device
        self.combine = nn.Linear(embed_dim, embed_dim)

    def _apply_quantum_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Run each token through the quantum layer and stack the results."""
        batch_size, seq_len, embed_dim = x.shape
        outputs = []
        for token in x.unbind(dim=1):
            # token shape: (batch, embed_dim)
            head_outputs = []
            for head in range(self.num_heads):
                # split token into sub‑vector for the head
                head_input = token[:, head * self.d_k : (head + 1) * self.d_k]
                # create or reuse quantum device
                qdev = self.q_device or tq.QuantumDevice(
                    n_wires=self.q_layer.n_wires, bsz=head_input.size(0), device=head_input.device
                )
                head_outputs.append(self.q_layer(head_input, qdev))
            outputs.append(torch.stack(head_outputs, dim=1))
        return torch.stack(outputs, dim=1)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:  # type: ignore[override]
        # Classical linear projections
        batch_size, seq_len, _ = x.shape
        q = self._apply_quantum_heads(x)
        k = self._apply_quantum_heads(x)
        v = self._apply_quantum_heads(x)
        # Scaled dot‑product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        out = torch.matmul(scores, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.combine(out)


class FeedForwardQuantum(nn.Module):
    """Quantum feed‑forward block that maps a token to a higher‑dimensional space."""

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

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            for wire, gate in enumerate(self.parameters):
                gate(q_device, wires=wire)
            return self.measure(q_device)

    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.q_layer = self.QLayer(n_qubits)
        self.q_device = tq.QuantumDevice(n_wires=n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        outputs = []
        for token in x.unbind(dim=1):
            qdev = self.q_device.copy(bsz=token.size(0), device=token.device)
            outputs.append(self.q_layer(token, qdev))
        out = torch.stack(outputs, dim=1)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))


class FeedForwardClassical(nn.Module):
    """Classical two‑layer MLP (used when quantum is disabled)."""

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerBlockQuantum(nn.Module):
    """Transformer block that can mix classical and quantum sub‑components."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        n_qubits_attn: int,
        n_qubits_ffn: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = (
            MultiHeadAttentionQuantum(embed_dim, num_heads, dropout, n_wires=n_qubits_attn)
            if n_qubits_attn > 0
            else MultiHeadAttentionQuantum(embed_dim, num_heads, dropout, n_wires=8)
        )
        self.ffn = (
            FeedForwardQuantum(embed_dim, ffn_dim, n_qubits_ffn, dropout)
            if n_qubits_ffn > 0
            else FeedForwardClassical(embed_dim, ffn_dim, dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class TransformerBlockClassical(nn.Module):
    """Purely classical transformer block (fallback)."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout, n_wires=8)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


# --------------------------------------------------------------------------- #
#  HybridTransformer – public API
# --------------------------------------------------------------------------- #
class HybridTransformer(nn.Module):
    """Hybrid transformer that can switch between classical and quantum blocks.

    Parameters
    ----------
    vocab_size : int
        Vocabulary size.
    embed_dim : int
        Embedding dimension.
    num_heads : int
        Number of attention heads.
    num_blocks : int
        Number of transformer blocks.
    ffn_dim : int
        Feed‑forward hidden dimension.
    num_classes : int
        Number of output classes.
    dropout : float, default 0.1
        Dropout probability.
    n_qubits_attn : int, default 0
        Number of qubits for the attention block. 0 → fully classical.
    n_qubits_ffn : int, default 0
        Number of qubits for the feed‑forward block. 0 → fully classical.
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
        n_qubits_attn: int = 0,
        n_qubits_ffn: int = 0,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim)
        blocks = [
            TransformerBlockQuantum(
                embed_dim, num_heads, ffn_dim, n_qubits_attn, n_qubits_ffn, dropout
            )
            if n_qubits_attn > 0 or n_qubits_ffn > 0
            else TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_blocks)
        ]
        self.transformer = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(
            embed_dim, num_classes if num_classes > 2 else 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        tokens = self.token_embedding(x)
        x = self.positional_encoding(tokens)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)


# --------------------------------------------------------------------------- #
#  Utility – regression helper (quantum version)
# --------------------------------------------------------------------------- #
def EstimatorQNN() -> QiskitEstimatorQNN:
    """Return a Qiskit EstimatorQNN that maps a 1‑qubit circuit to a scalar.

    The circuit consists of an H, an RX(θ) rotation on the input, and an RX(φ) weight
    that is trained.  The observable is the Pauli‑Y operator.
    """
    # Parameterised input and weight
    input_param = Parameter("θ")
    weight_param = Parameter("φ")

    qc = QuantumCircuit(1)
    qc.h(0)
    qc.ry(input_param, 0)
    qc.rx(weight_param, 0)

    # observable: Y
    observable = SparsePauliOp.from_list([("Y", 1)])

    estimator = StatevectorEstimator()
    estimator_qnn = QiskitEstimatorQNN(
        circuit=qc,
        observables=observable,
        input_params=[input_param],
        weight_params=[weight_param],
        estimator=estimator,
    )
    return estimator_qnn


# --------------------------------------------------------------------------- #
#  Backwards compatibility alias
# --------------------------------------------------------------------------- #
TextClassifier = HybridTransformer

__all__ = [
    "PositionalEncoding",
    "MultiHeadAttentionQuantum",
    "FeedForwardQuantum",
    "FeedForwardClassical",
    "TransformerBlockQuantum",
    "TransformerBlockClassical",
    "HybridTransformer",
    "TextClassifier",
    "EstimatorQNN",
]
