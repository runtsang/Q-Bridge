"""Quantum‑centric implementation of the hybrid transformer.

This module keeps the same public interface as the classical version but
adds quantum‑aware layers using TorchQuantum.  The quantum blocks are
designed to be drop‑in replacements for their classical counterparts.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.quantum as tq
import torch.quantum.functional as tqf


# --------------------------------------------------------------------------- #
#  Quantum attention
# --------------------------------------------------------------------------- #
class MultiHeadAttentionQuantum(tq.QuantumModule):
    """Multi‑head attention where each projection is encoded by a small circuit."""

    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int = 8) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
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

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.q_layer = self.QLayer()

    def _apply_quantum(self, x: torch.Tensor) -> torch.Tensor:
        """Apply quantum encoder to each token independently."""
        projections = []
        for token in x.unbind(dim=1):
            token = token.view(token.size(0), self.num_heads, -1)
            head_outputs = []
            for head in token.unbind(dim=1):
                qdev = tq.QuantumDevice(
                    n_wires=self.q_layer.n_wires, bsz=head.size(0), device=head.device
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
        # scaled dot‑product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        out = torch.matmul(scores, v)
        # reshape back
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        return out


# --------------------------------------------------------------------------- #
#  Quantum feed‑forward
# --------------------------------------------------------------------------- #
class FeedForwardQuantum(tq.QuantumModule):
    """Feed‑forward where the hidden layer is produced by a quantum circuit."""

    class QLayer(tq.QuantumModule):
        def __init__(self, n_qubits: int) -> None:
            super().__init__()
            self.n_qubits = n_qubits
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
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
            qdev = self.q_layer.qdev.copy(bsz=token.size(0), device=token.device)
            outputs.append(self.q_layer(token, qdev))
        out = torch.stack(outputs, dim=1)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))


# --------------------------------------------------------------------------- #
#  Transformer block
# --------------------------------------------------------------------------- #
class TransformerBlockQuantum(tq.QuantumModule):
    """Quantum transformer block that can mix classical and quantum sub‑modules."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        use_quantum_attn: bool = False,
        use_quantum_ffn: bool = False,
        n_qubits_attn: int = 8,
        n_qubits_ffn: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.attn = (
            MultiHeadAttentionQuantum(embed_dim, num_heads, dropout)
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
#  Positional encoding (identical to classical)
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
#  Quantum‑enhanced transformer
# --------------------------------------------------------------------------- #
class QuantumHybridTransformer(tq.QuantumModule):
    """Quantum‑aware transformer with optional classical/quantum blocks.

    Parameters
    ----------
    vocab_size : int
    embed_dim : int
    num_heads : int
    num_blocks : int
    ffn_dim : int
    num_classes : int
    dropout : float
    use_quantum_attn : bool
    use_quantum_ffn : bool
    n_qubits_attn : int
    n_qubits_ffn : int
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
#  Quantum regression support (same as classical seed but quantum)
# --------------------------------------------------------------------------- #
def generate_superposition_data(
    num_wires: int, samples: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate states of the form cos(theta)|0…0> + e^{i phi} sin(theta)|1…1>."""
    omega_0 = np.zeros(2 ** num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_wires, dtype=complex)
    omega_1[-1] = 1.0

    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels


class RegressionDataset(torch.utils.data.Dataset):
    """Dataset yielding quantum states and regression targets."""

    def __init__(self, samples: int, num_wires: int) -> None:
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class QuantumRegressionModel(tq.QuantumModule):
    """Quantum regression network used in the original seed."""

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
    "QuantumHybridTransformer",
    "generate_superposition_data",
    "RegressionDataset",
    "QuantumRegressionModel",
]
