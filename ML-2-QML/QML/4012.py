"""HybridTransformerQuantum: Quantum‑enhanced transformer.

This module implements a transformer that replaces the classical
attention and feed‑forward sub‑modules with parameterised quantum
circuits.  It is designed to be dropped into an existing training
pipeline that expects a PyTorch‑style ``nn.Module``.  The public API
mirrors the classical variant – passing ``use_quantum=True`` in the
constructor of the classical module redirects the user to this
implementation.

Key points

* The quantum blocks are implemented with torchquantum and expose
  the same ``forward`` interface as their classical counterparts.
* A small ``FastBaseEstimator`` is provided for evaluating a quantum
  circuit with a set of parameter values.  It uses
  ``qiskit.quantum_info.Statevector`` to obtain expectation values,
  matching the interface from the second reference pair.
* The module is self‑contained; it imports only the required
  dependencies and keeps the classical API identical to the original
  anchor reference.
"""

from __future__ import annotations

import math
from typing import Optional, Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator


# --------------------------------------------------------------------------- #
#  Classical building blocks reused for API compatibility
# --------------------------------------------------------------------------- #

class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float32) *
                             (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]

# --------------------------------------------------------------------------- #
#  Quantum‑aware attention block
# --------------------------------------------------------------------------- #

class MultiHeadAttentionQuantum(nn.Module):
    """Multi‑head attention where each head is parameterised by a small
    quantum circuit.  The circuit acts on a single token and produces
    a vector of size ``embed_dim``.  The output is concatenated across
    heads and projected back to ``embed_dim``."""
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

        # Each head has its own small circuit
        self.heads = nn.ModuleList([
            self._build_head(n_wires) for _ in range(num_heads)
        ])
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.q_device = q_device or tq.QuantumDevice(n_wires=n_wires)

    def _build_head(self, n_wires: int) -> tq.QuantumModule:
        class Head(tq.QuantumModule):
            def __init__(self):
                super().__init__()
                self.encoder = tq.GeneralEncoder(
                    [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
                )
                self.params = nn.ModuleList(
                    [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
                )
                self.measure = tq.MeasureAll(tq.PauliZ)

            def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
                self.encoder(q_device, x)
                for wire, gate in enumerate(self.params):
                    gate(q_device, wires=wire)
                return self.measure(q_device)

        return Head()

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        head_outputs = []
        for head in self.heads:
            # Prepare a quantum device per head for batch processing
            qdev = self.q_device.copy(bsz=batch, device=x.device)
            # Map token dimension to qubits: we treat each token as a vector
            # of length n_wires and feed it into the encoder.
            out = torch.stack(
                [head(token.unsqueeze(0), qdev) for token in x.unbind(dim=1)],
                dim=1
            )  # shape: (batch, seq_len, n_wires)
            # Project to d_k
            proj = nn.Linear(self.q_device.n_wires, self.d_k).to(x.device)
            head_outputs.append(proj(out))
        # Concatenate heads
        concat = torch.cat(head_outputs, dim=-1)  # (batch, seq_len, embed_dim)
        return self.out_proj(concat)

# --------------------------------------------------------------------------- #
#  Quantum feed‑forward block
# --------------------------------------------------------------------------- #

class FeedForwardQuantum(nn.Module):
    """Feed‑forward network realised by a small quantum circuit."""
    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

        self.q_layer = self._build_q_layer(n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def _build_q_layer(self, n_qubits: int) -> tq.QuantumModule:
        class QLayer(tq.QuantumModule):
            def __init__(self):
                super().__init__()
                self.encoder = tq.GeneralEncoder(
                    [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
                )
                self.params = nn.ModuleList(
                    [tq.RY(has_params=True, trainable=True) for _ in range(n_qubits)]
                )
                self.measure = tq.MeasureAll(tq.PauliZ)

            def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
                self.encoder(q_device, x)
                for wire, gate in enumerate(self.params):
                    gate(q_device, wires=wire)
                return self.measure(q_device)
        return QLayer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        outputs = []
        for token in x.unbind(dim=1):
            qdev = self.q_layer.q_device.copy(bsz=1, device=token.device)
            out = self.q_layer(token.unsqueeze(0), qdev)
            outputs.append(out)
        out = torch.stack(outputs, dim=1)  # (batch, seq_len, n_qubits)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))

# --------------------------------------------------------------------------- #
#  Transformer block
# --------------------------------------------------------------------------- #

class TransformerBlockQuantum(nn.Module):
    """Transformer block that uses quantum attention and feed‑forward."""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        n_wires: int = 8,
        n_qubits_ffn: int = 8,
        dropout: float = 0.1,
        q_device: Optional[tq.QuantumDevice] = None,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout, n_wires, q_device)
        self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits_ffn, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

# --------------------------------------------------------------------------- #
#  Fast estimator for quantum circuits
# --------------------------------------------------------------------------- #

class FastBaseEstimator:
    """Evaluate expectation values of observables for a parameterised
    quantum circuit.  Mirrors the implementation from the second
    reference pair but is integrated into this module."""
    def __init__(self, circuit: QuantumCircuit):
        self.circuit = circuit
        self.parameters = list(circuit.parameters)

    def _bind(self, param_values: Sequence[float]) -> QuantumCircuit:
        if len(param_values)!= len(self.parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self.parameters, param_values))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Return expectation values for each observable and parameter set."""
        observables = list(observables)
        results: List[List[complex]] = []
        for params in parameter_sets:
            circ = self._bind(params)
            state = Statevector.from_instruction(circ)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

# --------------------------------------------------------------------------- #
#  Hybrid quantum transformer
# --------------------------------------------------------------------------- #

class HybridTransformer(nn.Module):
    """
    Quantum‑enhanced transformer that mirrors the classical API.  The
    constructor accepts the same arguments as the classical variant
    but automatically replaces the attention and feed‑forward
    sub‑modules with their quantum counterparts.
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
        n_wires: int = 8,
        n_qubits_ffn: int = 8,
        q_device: Optional[tq.QuantumDevice] = None,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)

        q_device = q_device or tq.QuantumDevice(n_wires=max(n_wires, n_qubits_ffn))
        blocks = [
            TransformerBlockQuantum(
                embed_dim, num_heads, ffn_dim,
                n_wires=n_wires,
                n_qubits_ffn=n_qubits_ffn,
                q_device=q_device,
                dropout=dropout,
            )
            for _ in range(num_blocks)
        ]
        self.blocks = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(x)
        x = self.pos_encoder(x)
        x = self.blocks(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)

__all__ = [
    "PositionalEncoder",
    "MultiHeadAttentionQuantum",
    "FeedForwardQuantum",
    "TransformerBlockQuantum",
    "FastBaseEstimator",
    "HybridTransformer",
]
