"""HybridEstimator: quantum estimator with quantum transformer.

The module defines a hybrid estimator that uses a torchquantum‑based
quantum transformer to evaluate expectation values of observables.
It can be used directly with a quantum circuit or a quantum
transformer model.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List

import math
import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
from qiskit.quantum_info.operators.base_operator import BaseOperator

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


class MultiHeadAttentionQuantum(tq.QuantumModule):
    """Quantum multi‑head attention module."""

    def __init__(self, embed_dim: int, num_heads: int, n_wires: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
        )
        self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        outputs = []
        for i in range(seq_len):
            token = x[:, i, :].view(batch, self.num_heads, self.d_k)
            head_outputs = []
            for head in token.unbind(dim=1):
                qdev = q_device.copy(bsz=batch, device=head.device)
                self.encoder(qdev, head)
                for wire, gate in enumerate(self.params):
                    gate(qdev, wires=wire)
                head_outputs.append(self.measure(qdev))
            outputs.append(torch.stack(head_outputs, dim=1))
        return torch.stack(outputs, dim=1)


class FeedForwardQuantum(tq.QuantumModule):
    """Quantum feed‑forward block."""

    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.n_qubits = n_qubits
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
        )
        self.params = nn.ModuleList([tq.RY(has_params=True, trainable=True) for _ in range(n_qubits)])
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        outputs = []
        for i in range(seq_len):
            token = x[:, i, :].view(batch, self.n_qubits)
            qdev = q_device.copy(bsz=batch, device=token.device)
            self.encoder(qdev, token)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            meas = self.measure(qdev)
            outputs.append(meas)
        out = torch.stack(outputs, dim=1)
        out = self.linear1(out)
        out = self.linear2(out)
        return out


class TransformerBlockQuantum(tq.QuantumModule):
    """Quantum transformer block."""

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, n_qubits: int = 8):
        super().__init__()
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, n_qubits)
        self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
        attn_out = self.attn(x, q_device)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x, q_device)
        return self.norm2(x + self.dropout(ffn_out))


class PositionalEncoder(tq.QuantumModule):
    """Sinusoidal positional encoding."""

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


class TextClassifierQuantum(tq.QuantumModule):
    """Quantum transformer‑based text classifier."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        n_qubits: int = 8,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)
        self.blocks = nn.ModuleList(
            [TransformerBlockQuantum(embed_dim, num_heads, ffn_dim, n_qubits) for _ in range(num_blocks)]
        )
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_embedding(tokens)
        for block in self.blocks:
            x = block(x, q_device)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)


class HybridEstimator:
    """Quantum estimator that uses a quantum transformer.

    Parameters
    ----------
    model : tq.QuantumModule
        A quantum transformer model (e.g. TextClassifierQuantum) that
        can be used to transform inputs and produce a state vector.
    observables : Iterable[BaseOperator]
        Quantum observables to evaluate on the output state.
    q_device : tq.QuantumDevice, optional
        Quantum device used for simulation.  If None, a default
        device is created.
    """

    def __init__(
        self,
        model: tq.QuantumModule,
        observables: Iterable[BaseOperator],
        *,
        q_device: tq.QuantumDevice | None = None,
    ) -> None:
        self.model = model
        self.observables = list(observables)
        self.q_device = q_device or tq.QuantumDevice(n_wires=model.n_wires)

    def evaluate(
        self,
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Evaluate the quantum model for each parameter set.

        The parameters are bound to the model's trainable gates
        (e.g. RX, RY) before the forward pass.
        """
        results: List[List[complex]] = []
        for params in parameter_sets:
            # Bind parameters to the model
            self.model.set_parameters(params)
            # Run the model to obtain a statevector
            state = self.model.forward(self.q_device)
            # Compute expectation values for each observable
            row = [state.expectation_value(obs) for obs in self.observables]
            results.append(row)
        return results
