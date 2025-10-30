"""Hybrid transformer and regression framework combining classical and quantum layers.

The module preserves the public API of the original QTransformerTorch.py
while adding the following extensions:

* ``HybridTransformer`` – a transformer that can be configured to use
  classical, quantum, or a mixture of attention and feed‑forward modules.
* ``QuantumRegressionModel`` – a TorchQuantum based regression network.
* ``generate_superposition_data`` – generates data in both classical
  (float32) and quantum (complex64) formats.
* ``RegressionDataset`` – dataset wrapper that can yield either classical
  or quantum states.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


# --------------------------------------------------------------------------- #
#  Core attention and feed‑forward primitives
# --------------------------------------------------------------------------- #
class MultiHeadAttentionBase(nn.Module):
    """Shared logic for all attention variants."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape to (batch, heads, seq, d_k)."""
        batch_size = x.size(0)
        return (
            x.view(batch_size, -1, self.num_heads, self.d_k)
           .transpose(1, 2)
           .contiguous()
        )

    def attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Scaled dot‑product attention."""
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
        out, _ = self.attention(q, k, v, mask)
        return out.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)


class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    """Standard multi‑head attention implemented with linear layers."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        use_bias: bool = False,
    ) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.combine_heads = nn.Linear(embed_dim, embed_dim, bias=use_bias)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, _, embed_dim = x.size()
        if embed_dim!= self.embed_dim:
            raise ValueError(
                f"Input embedding ({embed_dim}) does not match layer embedding size ({self.embed_dim})"
            )
        k = self.k_linear(x)
        q = self.q_linear(x)
        v = self.v_linear(x)
        out = self.downstream(q, k, v, batch_size, mask)
        return self.combine_heads(out)


class MultiHeadAttentionQuantum(MultiHeadAttentionBase):
    """Attention where each projection is processed by a small quantum circuit."""

    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int = 8) -> None:
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

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        use_bias: bool = False,
        q_device: Optional[tq.QuantumDevice] = None,
    ) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.q_layer = self.QLayer()
        self.q_device = q_device
        self.combine_heads = nn.Linear(embed_dim, embed_dim, bias=use_bias)

    def _apply_quantum(self, x: torch.Tensor) -> torch.Tensor:
        """Apply quantum encoder to each token independently."""
        projections = []
        for token in x.unbind(dim=1):
            token = token.view(token.size(0), self.num_heads, -1)
            head_outputs = []
            for head in token.unbind(dim=1):
                qdev = (
                    self.q_device
                    or tq.QuantumDevice(
                        n_wires=self.q_layer.n_wires,
                        bsz=head.size(0),
                        device=head.device,
                    )
                )
                head_outputs.append(self.q_layer(head, qdev))
            projections.append(torch.stack(head_outputs, dim=1))
        return torch.stack(projections, dim=1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, _, embed_dim = x.size()
        if embed_dim!= self.embed_dim:
            raise ValueError(
                f"Input embedding ({embed_dim}) does not match layer embedding size ({self.embed_dim})"
            )
        k = self._apply_quantum(x)
        q = self._apply_quantum(x)
        v = self._apply_quantum(x)
        out = self.downstream(q, k, v, batch_size, mask)
        return self.combine_heads(out)


class FeedForwardBase(nn.Module):
    """Base class for feed‑forward modules."""

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)


class FeedForwardClassical(FeedForwardBase):
    """Standard two‑layer MLP."""

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class FeedForwardQuantum(FeedForwardBase):
    """Feed‑forward where the hidden layer is produced by a quantum circuit."""

    class QLayer(tq.QuantumModule):
        def __init__(self, n_qubits: int) -> None:
            super().__init__()
            self.n_qubits = n_qubits
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [i], "func": "rx", "wires": [i]}
                    for i in range(n_qubits)
                ]
            )
            self.params = nn.ModuleList(
                [tq.RY(has_params=True, trainable=True) for _ in range(n_qubits)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(qdev, x)
            for gate, wire in zip(self.params, range(self.n_qubits)):
                gate(qdev, wires=wire)
            return self.measure(qdev)

    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.q_layer = self.QLayer(n_qubits)
        self.q_device = tq.QuantumDevice(n_wires=n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for token in x.unbind(dim=1):
            qdev = self.q_device.copy(bsz=token.size(0), device=token.device)
            outputs.append(self.q_layer(token, qdev))
        out = torch.stack(outputs, dim=1)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))


# --------------------------------------------------------------------------- #
#  Transformer block
# --------------------------------------------------------------------------- #
class TransformerBlockBase(nn.Module):
    """Base class for a transformer block."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class TransformerBlockClassical(TransformerBlockBase):
    """Fully classical transformer block."""

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
    """Transformer block that can mix classical and quantum sub‑modules."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        use_quantum_attn: bool = False,
        use_quantum_ffn: bool = False,
        n_qubits_attn: int = 8,
        n_qubits_ffn: int = 8,
        q_device: Optional[tq.QuantumDevice] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = (
            MultiHeadAttentionQuantum(
                embed_dim, num_heads, dropout, q_device=q_device
            )
            if use_quantum_attn
            else MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        )
        self.ffn = (
            FeedForwardQuantum(embed_dim, ffn_dim, n_qubits_ffn, dropout)
            if use_quantum_ffn
            else FeedForwardClassical(embed_dim, ffn_dim, dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


# --------------------------------------------------------------------------- #
#  Positional encoding
# --------------------------------------------------------------------------- #
class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


# --------------------------------------------------------------------------- #
#  Hybrid transformer model
# --------------------------------------------------------------------------- #
class HybridTransformer(nn.Module):
    """Transformer that can be configured to use classical or quantum layers.

    Parameters
    ----------
    vocab_size : int
        Size of the token vocabulary.
    embed_dim : int
        Embedding dimension.
    num_heads : int
        Number of attention heads.
    num_blocks : int
        Number of transformer blocks.
    ffn_dim : int
        Hidden dimension of the feed‑forward sub‑module.
    num_classes : int
        Number of output classes. 1 signals regression.
    dropout : float
        Drop‑out probability.
    use_quantum_attn : bool
        If True, all attention layers are quantum.
    use_quantum_ffn : bool
        If True, all feed‑forward layers are quantum.
    n_qubits_attn : int
        Number of qubits per attention head.
    n_qubits_ffn : int
        Number of qubits in the feed‑forward quantum module.
    q_device : Optional[tq.QuantumDevice]
        Shared quantum device.
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
        use_quantum_attn: bool = False,
        use_quantum_ffn: bool = False,
        n_qubits_attn: int = 8,
        n_qubits_ffn: int = 8,
        q_device: Optional[tq.QuantumDevice] = None,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)
        blocks: List[nn.Module] = [
            TransformerBlockQuantum(
                embed_dim,
                num_heads,
                ffn_dim,
                use_quantum_attn=use_quantum_attn,
                use_quantum_ffn=use_quantum_ffn,
                n_qubits_attn=n_qubits_attn,
                n_qubits_ffn=n_qubits_ffn,
                q_device=q_device,
                dropout=dropout,
            )
            for _ in range(num_blocks)
        ]
        self.transformers = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(dropout)
        self.classifier = (
            nn.Linear(embed_dim, num_classes) if num_classes > 1 else nn.Linear(embed_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_embedding(tokens)
        x = self.transformers(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)


# --------------------------------------------------------------------------- #
#  Quantum regression support
# --------------------------------------------------------------------------- #
def generate_superposition_data(
    num_features: int, samples: int, *, quantum: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate states of the form cos(theta)|0…0> + e^{i phi} sin(theta)|1…1>.

    Parameters
    ----------
    num_features : int
        Number of qubits (if quantum) or classical features.
    samples : int
        Number of samples to generate.
    quantum : bool, optional
        If True, return complex64 states suitable for TorchQuantum.
        Otherwise, return real32 features.

    Returns
    -------
    states : np.ndarray
        Shape (samples, 2**num_features) if quantum, else (samples, num_features).
    labels : np.ndarray
        Regression target.
    """
    if quantum:
        omega_0 = np.zeros(2 ** num_features, dtype=complex)
        omega_0[0] = 1.0
        omega_1 = np.zeros(2 ** num_features, dtype=complex)
        omega_1[-1] = 1.0

        thetas = 2 * np.pi * np.random.rand(samples)
        phis = 2 * np.pi * np.random.rand(samples)
        states = np.zeros((samples, 2 ** num_features), dtype=complex)
        for i in range(samples):
            states[i] = (
                np.cos(thetas[i]) * omega_0
                + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
            )
        labels = np.sin(2 * thetas) * np.cos(phis)
    else:
        x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
        angles = x.sum(axis=1)
        labels = np.sin(angles) + 0.1 * np.cos(2 * angles)
        states = x
    return states.astype(np.complex64 if quantum else np.float32), labels.astype(np.float32)


class RegressionDataset(torch.utils.data.Dataset):
    """Dataset yielding either classical or quantum states."""

    def __init__(self, samples: int, num_features: int, *, quantum: bool = False) -> None:
        self.states, self.labels = generate_superposition_data(num_features, samples, quantum=quantum)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat if self.states.dtype == np.complex64 else torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class QuantumRegressionModel(tq.QuantumModule):
    """Quantum‑enhanced regression network."""

    class QLayer(tq.QuantumModule):
        def __init__(self, num_wires: int) -> None:
            super().__init__()
            self.n_wires = num_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random_layer(qdev)
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)

    def __init__(self, num_wires: int) -> None:
        super().__init__()
        self.n_wires = num_wires
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict[f"{num_wires}xRy"]
        )
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
    "HybridTransformer",
    "generate_superposition_data",
    "RegressionDataset",
    "QuantumRegressionModel",
]
