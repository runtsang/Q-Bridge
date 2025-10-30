"""Unified quantum estimator with quantum modules and shot‑noise.

This module re‑implements the quantum side of the reference pairs:
- FastBaseEstimator evaluation via qiskit Statevector.
- FastEstimator shot‑noise via AerSimulator.
- QuantumRegression encoder and dataset.
- Quanvolution filter using TorchQuantum.
- Quantum transformer blocks using TorchQuantum.

The UnifiedEstimator class accepts either a qiskit QuantumCircuit or a
tq.QuantumModule.  For circuits it evaluates expectation values of
BaseOperator observables; for modules it forwards the inputs and applies
callable observables.  Shot‑noise is simulated with AerSimulator when
shots is provided.
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional, Union

import qiskit
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.providers.aer import AerSimulator

import torchquantum as tq
import torchquantum.functional as tqf

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


# ---------- Data generation ---------------------------------------------

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a set of superposition states and labels."""
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
    return states, labels.astype(np.float32)


# ---------- Quantum encoder ---------------------------------------------

class QuantumEncoder(tq.QuantumModule):
    """Variational encoder mapping a classical vector into a quantum state."""

    def __init__(self, n_wires: int):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(n_wires)]
        )
        self.random = tq.RandomLayer(n_ops=20, wires=list(range(n_wires)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)

    def forward(self, qdev: tq.QuantumDevice) -> None:  # pragma: no cover
        self.random(qdev)
        for wire in range(self.n_wires):
            self.rx(qdev, wires=wire)
            self.ry(qdev, wires=wire)


class QuantumRegressionModel(tq.QuantumModule):
    """Quantum regression model mirroring the TorchQuantum example."""

    def __init__(self, n_wires: int):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = QuantumEncoder(n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(n_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)


# ---------- Quanvolution filter ----------------------------------------

class QuanvolutionFilter(tq.QuantumModule):
    """Apply a random two‑qubit quantum kernel to 2×2 image patches."""

    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                data = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, data)
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)


class QuanvolutionClassifier(nn.Module):
    """Hybrid classifier using the quantum filter."""

    def __init__(self) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)


# ---------- Quantum transformer block ----------------------------------

class MultiHeadAttentionQuantum(tq.QuantumModule):
    """Quantum multi‑head attention head."""

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

        def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.parameters):
                gate(qdev, wires=wire)
            for wire in range(self.n_wires - 1):
                tqf.cnot(qdev, wires=[wire, wire + 1])
            tqf.cnot(qdev, wires=[self.n_wires - 1, 0])
            return self.measure(qdev)

    def __init__(self, embed_dim: int, num_heads: int, n_qubits: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.n_qubits = n_qubits
        self.heads = nn.ModuleList([self.QLayer(n_qubits) for _ in range(num_heads)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        # Simplified: process each token independently
        qdev = tq.QuantumDevice(n_wires=self.n_qubits, bsz=batch, device=x.device)
        outputs = []
        for token in x.unbind(dim=1):
            token = token.unsqueeze(0)  # (1, embed_dim)
            head_outs = []
            for head in self.heads:
                head_outs.append(head(token, qdev))
            outputs.append(torch.stack(head_outs, dim=1))
        return torch.stack(outputs, dim=1)


class FeedForwardQuantum(tq.QuantumModule):
    """Quantum feed‑forward network."""

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

        def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.parameters):
                gate(qdev, wires=wire)
            return self.measure(qdev)

    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.n_qubits = n_qubits
        self.q_layer = self.QLayer(n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_qubits, bsz=batch, device=x.device)
        out = self.q_layer(x, qdev)
        out = self.linear1(F.dropout(out, p=0.1))
        return self.linear2(F.relu(out))


class TransformerBlockQuantum(tq.QuantumModule):
    """Quantum transformer block combining attention and feed‑forward."""

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, n_qubits: int = 8):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, n_qubits)
        self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding used by the transformer."""

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


class TextClassifierQuantum(nn.Module):
    """Transformer‑based classifier with quantum attention and feed‑forward."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        n_qubits: int = 8,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)
        self.blocks = nn.ModuleList(
            [
                TransformerBlockQuantum(embed_dim, num_heads, ffn_dim, n_qubits)
                for _ in range(num_blocks)
            ]
        )
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_embedding(tokens)
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)


# ---------- Unified estimator -----------------------------------------

class UnifiedEstimator:
    """Quantum estimator that can wrap a qiskit circuit or a TorchQuantum module.

    The evaluate method returns a list of lists of complex numbers.  If a
    circuit is provided, expectation values of BaseOperator observables are
    computed with Statevector (exact) or AerSimulator (shots).  If a
    TorchQuantum module is provided, the module's forward is called and the
    resulting tensor is fed through the supplied observables.
    """

    def __init__(
        self,
        model: Union[QuantumCircuit, tq.QuantumModule],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.model = model
        self.shots = shots
        self.seed = seed
        if isinstance(model, QuantumCircuit):
            self._backend = AerSimulator(method="statevector") if shots is None else AerSimulator(method="qasm")
        else:
            self._backend = None

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if not isinstance(self.model, QuantumCircuit):
            raise TypeError("Binding is only supported for QuantumCircuit models.")
        if len(parameter_values)!= len(self.model.parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self.model.parameters, parameter_values))
        return self.model.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        if isinstance(self.model, QuantumCircuit):
            return self._evaluate_circuit(observables, parameter_sets)
        else:
            return self._evaluate_module(observables, parameter_sets)

    def _evaluate_circuit(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        results: List[List[complex]] = []
        for params in parameter_sets:
            bound_circ = self._bind(params)
            if self.shots is None:
                state = Statevector.from_instruction(bound_circ)
                row = [state.expectation_value(obs) for obs in observables]
            else:
                bound_circ.save_statevector()
                self._backend.set_options(seed_simulator=self.seed, seed_transpiler=self.seed)
                job = self._backend.run(bound_circ, shots=self.shots)
                result = job.result()
                state = result.get_statevector(bound_circ)
                row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

    def _evaluate_module(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        results: List[List[complex]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.model(inputs)
                row: List[complex] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        val = val.mean().item()
                    row.append(complex(val))
                results.append(row)
        return results
