"""Quantum‑centric hybrid estimator and transformer.

The QML module mirrors the deterministic FastBaseEstimator logic but
adds a quantum‑parameterized circuit that can be bound with a
parameter set.  The estimator can also evaluate quantum kernels as
observables.  The code builds on the reference QuantumKernelMethod
and QTransformerTorch modules, merging them into a single, reusable
module that can be imported by a quick‑start script.
"""

from __future__ import annotations

import math
import numpy as np
import torch
from torch import nn
from collections.abc import Iterable, Sequence
from typing import List

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator

import torchquantum as tq
from torchquantum.functional import func_name_dict, op_name_dict
import torchquantum.functional as tqf

# --------------------------------------------------------------------------- #
# Quantum estimator
# --------------------------------------------------------------------------- #
class QuantumEstimator:
    """Quantum circuit to bind parameters and return statevector."""
    def __init__(self, circuit: QuantumCircuit):
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, param_values: Sequence[float]) -> QuantumCircuit:
        if len(param_values)!= len(self._parameters):
            raise ValueError(f"Expected {len(self._parameters)} parameters, got {len(param_values)}")
        mapping = dict(zip(self._parameters, param_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        param_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        results: List[List[complex]] = []
        for params in param_sets:
            qc = self._bind(params)
            state = Statevector.from_instruction(qc)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

# --------------------------------------------------------------------------- #
# Quantum kernel
# --------------------------------------------------------------------------- #
class QuantumKernel(tq.QuantumModule):
    """Encodes classical data through a programmable list of quantum gates."""
    def __init__(self, func_list):
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = x[:, info["input_idx"]] if op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]] if op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

class Kernel(tq.QuantumModule):
    """Quantum kernel evaluated via a fixed TorchQuantum ansatz."""
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = QuantumKernel(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    kernel = Kernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# --------------------------------------------------------------------------- #
# Simple quantum attention and feed‑forward
# --------------------------------------------------------------------------- #
class QuantumAttention(tq.QuantumModule):
    """Quantum multi‑head attention using a simple parameterised circuit."""
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.n_wires = num_heads * self.head_dim
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(self.n_wires)]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, embed_dim)
        batch_size = x.shape[0]
        x = x.view(batch_size, self.num_heads, self.head_dim)
        outputs = []
        for token in x.unbind(dim=1):
            qdev = self.q_device.copy(bsz=batch_size)
            self.encoder(qdev, token)
            out = self.measure(qdev)
            outputs.append(out)
        return torch.stack(outputs, dim=1).reshape(batch_size, self.embed_dim)

class QuantumFeedForward(tq.QuantumModule):
    """Quantum feed‑forward using a small circuit."""
    def __init__(self, embed_dim: int, ffn_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.n_wires = embed_dim
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(self.n_wires)]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.linear = nn.Linear(self.n_wires, self.ffn_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        qdev = self.q_device.copy(bsz=batch_size)
        self.encoder(qdev, x)
        out = self.measure(qdev)
        return self.linear(out)

# --------------------------------------------------------------------------- #
# Positional encoding
# --------------------------------------------------------------------------- #
class PositionalEncoder(nn.Module):
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

# --------------------------------------------------------------------------- #
# Quantum text classifier
# --------------------------------------------------------------------------- #
class QuantumTextClassifier(nn.Module):
    """Transformer‑based text classifier supporting quantum submodules."""
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    QuantumAttention(embed_dim, num_heads),
                    nn.Dropout(dropout),
                    QuantumFeedForward(embed_dim, ffn_dim),
                    nn.Dropout(dropout),
                )
                for _ in range(num_blocks)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_embedding(tokens)
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)

# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
__all__ = [
    "QuantumEstimator",
    "QuantumKernel",
    "kernel_matrix",
    "QuantumAttention",
    "QuantumFeedForward",
    "PositionalEncoder",
    "QuantumTextClassifier",
]
