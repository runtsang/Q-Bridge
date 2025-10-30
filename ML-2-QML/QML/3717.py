"""Quantum‑enhanced transformer with variational circuits and fast expectation evaluation."""

from __future__ import annotations

import math
import numpy as np
from typing import Callable, List, Sequence, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


# ---------- Quantum modules ----------
class QuantumAttention(tq.QuantumModule):
    """Multi‑head attention realised through a variational quantum circuit."""

    class QLayer(tq.QuantumModule):
        def __init__(self) -> None:
            super().__init__()
            self.n_wires = 8
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(self.n_wires)]
            )
            self.parameters = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(self.n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            for gate in self.parameters:
                gate(q_device)
            return self.measure(q_device)

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.q_layer = self.QLayer()
        self.combine = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, seq, _ = x.shape
        x_proj = x.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        out = []
        for head in torch.unbind(x_proj, dim=1):
            q_device = tq.QuantumDevice(n_wires=self.q_layer.n_wires, bsz=head.size(0), device=head.device)
            out.append(self.q_layer(head, q_device))
        out = torch.stack(out, dim=1).transpose(1, 2).contiguous().view(batch, seq, -1)
        attn_out, _ = self.attn(out, out, out, key_padding_mask=mask)
        return self.combine(attn_out)


class QuantumFeedForward(tq.QuantumModule):
    """Feed‑forward network implemented as a variational quantum circuit."""

    class QLayer(tq.QuantumModule):
        def __init__(self, n_qubits: int) -> None:
            super().__init__()
            self.n_qubits = n_qubits
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
            )
            self.parameters = nn.ModuleList(
                [tq.RY(has_params=True, trainable=True) for _ in range(n_qubits)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            for gate in self.parameters:
                gate(q_device)
            return self.measure(q_device)

    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.q_layer = self.QLayer(n_qubits)
        self.input_proj = nn.Linear(embed_dim, n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = []
        for token in torch.unbind(x, dim=1):  # token shape (batch, embed_dim)
            token_proj = self.input_proj(token)  # (batch, n_qubits)
            q_device = tq.QuantumDevice(n_wires=self.n_qubits, bsz=token_proj.size(0), device=token_proj.device)
            out.append(self.q_layer(token_proj, q_device))
        out = torch.stack(out, dim=1)  # (batch, seq, n_qubits)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))


# ---------- Transformer blocks ----------
class TransformerBlockQuantum(nn.Module):
    """
    Transformer block that stitches together quantum attention and quantum feed‑forward.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        n_qubits_ffn: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.attn = QuantumAttention(embed_dim, num_heads, dropout)
        self.ffn = QuantumFeedForward(embed_dim, ffn_dim, n_qubits_ffn, dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_out = self.attn(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class TransformerBlockClassical(nn.Module):
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
        self.dropout = nn.Dropout(dropout)

        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(F.relu(self.linear2(self.ffn(x)))))
        return x


# ---------- Positional encoding ----------
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


# ---------- Hybrid transformer ----------
class HybridTransformer(nn.Module):
    """
    Quantum‑enhanced transformer that optionally falls back to classical layers.
    Extends the original QTransformerTorch with a variational circuit interface and
    a fast expectation evaluator inspired by FastEstimator.
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
        use_quantum: bool = False,
        n_qubits_ffn: int = 8,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional = PositionalEncoder(embed_dim)
        self.blocks = nn.Sequential(
            *[
                (
                    TransformerBlockQuantum
                    if use_quantum
                    else TransformerBlockClassical
                )(
                    embed_dim,
                    num_heads,
                    ffn_dim,
                    n_qubits_ffn if use_quantum else 0,
                    dropout,
                )
                for _ in range(num_blocks)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(
            embed_dim, num_classes if num_classes > 2 else 1
        )

    # ---------- Forward ----------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.positional(tokens)
        x = self.blocks(x)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)

    # ---------- Evaluation ----------
    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[int]],
    ) -> List[List[float]]:
        """
        Fast evaluation of the transformer on a list of input sequences.
        Observables are callables applied to the logits.
        """
        self.eval()
        observables = list(observables) or [lambda logits: logits.mean(dim=-1)]
        results: List[List[float]] = []
        with torch.no_grad():
            for params in parameter_sets:
                batch = torch.as_tensor(params, dtype=torch.long)
                if batch.ndim == 1:
                    batch = batch.unsqueeze(0)
                logits = self(batch)
                row: List[float] = []
                for obs in observables:
                    val = obs(logits)
                    if isinstance(val, torch.Tensor):
                        val = val.mean().item()
                    row.append(float(val))
                results.append(row)
        return results

    def evaluate_with_shots(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[int]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """
        Adds Gaussian shot noise to the evaluation, mirroring FastEstimator.
        """
        base = self.evaluate(observables, parameter_sets)
        if shots is None:
            return base
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in base:
            noisy_row = [
                float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row
            ]
            noisy.append(noisy_row)
        return noisy

    # ---------- Variational circuit helper ----------
    def to_qiskit_circuit(self, token_sequence: Sequence[int]) -> QuantumCircuit:
        """
        Convert a single token sequence into a parameter‑ised Qiskit circuit.
        The circuit consists of RX rotations per token followed by a variational layer
        mimicking the quantum attention block.
        """
        qc = QuantumCircuit(len(token_sequence))
        for idx, token in enumerate(token_sequence):
            qc.rx(token * 0.01, idx)  # encode token as small rotation
        # Add a simple variational layer
        for i in range(len(token_sequence) - 1):
            qc.cx(i, i + 1)
        qc.barrier()
        return qc

    # ---------- Expectation of a quantum circuit ----------
    def quantum_expectation(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[int]],
    ) -> List[List[complex]]:
        """
        Compute expectation values for a quantum circuit equivalent to the transformer
        for each set of token parameters.  Uses Qiskit Statevector for exact simulation.
        """
        results: List[List[complex]] = []
        for params in parameter_sets:
            qc = self.to_qiskit_circuit(params)
            state = Statevector.from_instruction(qc)
            row = [state.expectation_value(op) for op in observables]
            results.append(row)
        return results


__all__ = [
    "HybridTransformer",
]
