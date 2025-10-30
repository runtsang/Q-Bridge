"""Quantum‑enhanced transformer classifier with a quantum expectation head.

The module mirrors the classical implementation but replaces the
attention, feed‑forward, and final hybrid head with quantum‑enabled
sub‑modules.  The API remains identical, enabling drop‑in
replacement for the classical version.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

import qiskit
from qiskit import assemble, transpile
import numpy as np


class QuantumCircuit:
    """Parameterized two‑qubit circuit returning an expectation value."""

    def __init__(self, n_qubits: int, backend, shots: int) -> None:
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        all_qubits = list(range(n_qubits))
        self.theta = qiskit.circuit.Parameter("theta")

        self._circuit.h(all_qubits)
        self._circuit.barrier()
        self._circuit.ry(self.theta, all_qubits)
        self._circuit.measure_all()

        self.backend = backend
        self.shots = shots

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self._circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()

        def expectation(count_dict):
            counts = np.array(list(count_dict.values()))
            states = np.array(list(count_dict.keys())).astype(float)
            probabilities = counts / self.shots
            return np.sum(states * probabilities)

        if isinstance(result, list):
            return np.array([expectation(item) for item in result])
        return np.array([expectation(result)])


class HybridFunction(torch.autograd.Function):
    """Differentiable interface that feeds activations through a quantum circuit."""

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        expectation = ctx.circuit.run(inputs.tolist())
        result = torch.tensor([expectation])
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = np.ones_like(inputs.tolist()) * ctx.shift
        grads = []
        for idx, val in enumerate(inputs.tolist()):
            grads.append(
                ctx.circuit.run([val + shift[idx]]) - ctx.circuit.run([val - shift[idx]])
            )
        grads = torch.tensor([grads]).float()
        return grads * grad_output.float(), None, None


class Hybrid(nn.Module):
    """Quantum‑based hybrid head that replaces a classical linear layer."""

    def __init__(self, n_qubits: int, backend, shots: int, shift: float = np.pi / 2) -> None:
        super().__init__()
        self.circuit = QuantumCircuit(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs, self.circuit, self.shift)


class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding identical to the classical version."""

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


class MultiHeadAttentionQuantum(tq.QuantumModule):
    """Quantum‑enabled multi‑head attention via a parameterized quantum module."""

    class QLayer(tq.QuantumModule):
        def __init__(self) -> None:
            super().__init__()
            self.n_wires = 8
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [i], "func": "rx", "wires": [i]}
                    for i in range(self.n_wires)
                ]
            )
            self.parameters = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(self.n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            for wire, gate in enumerate(self.parameters):
                gate(q_device, wires=wire)
            for wire in range(self.n_wires - 1):
                tqf.cnot(q_device, wires=[wire, wire + 1])
            tqf.cnot(q_device, wires=[self.n_wires - 1, 0])
            return self.measure(q_device)

    def __init__(self, embed_dim: int, num_heads: int, q_device: Optional[tq.QuantumDevice] = None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.q_layer = self.QLayer()
        self.q_device = q_device or tq.QuantumDevice(n_wires=self.q_layer.n_wires)
        self.combine = nn.Linear(embed_dim, embed_dim)

    def _apply_quantum(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        seq_len = x.size(1)
        heads = []
        for t in x.unbind(dim=1):
            t = t.view(batch_size, self.num_heads, self.d_k)
            outputs = []
            for h in t.unbind(dim=1):
                qdev = self.q_device.copy(bsz=batch_size, device=h.device)
                outputs.append(self.q_layer(h, qdev))
            heads.append(torch.stack(outputs, dim=1))
        return torch.stack(heads, dim=1).view(batch_size, seq_len, self.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._apply_quantum(x)


class FeedForwardQuantum(tq.QuantumModule):
    """Quantum‑enabled feed‑forward network."""

    class QLayer(tq.QuantumModule):
        def __init__(self, n_qubits: int) -> None:
            super().__init__()
            self.n_wires = n_qubits
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
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

    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.q_layer = self.QLayer(n_qubits)
        self.q_device = tq.QuantumDevice(n_wires=n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for t in x.unbind(dim=1):
            qdev = self.q_device.copy(bsz=t.size(0), device=t.device)
            outputs.append(self.q_layer(t, qdev))
        out = torch.stack(outputs, dim=1)
        out = self.linear1(out)
        return self.linear2(F.relu(out))


class TransformerBlockQuantum(nn.Module):
    """Full transformer block with quantum attention and feed‑forward."""

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
        self.dropout = nn.Dropout(dropout)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads)
        self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits_ffn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class HybridTransformerClassifier(nn.Module):
    """Quantum‑enhanced transformer classifier mirroring the classical API."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int = 2,
        dropout: float = 0.1,
        n_qubits_transformer: int = 8,
        n_qubits_ffn: int = 8,
        shift: float = np.pi / 2,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)
        self.transformers = nn.Sequential(
            *[
                TransformerBlockQuantum(
                    embed_dim,
                    num_heads,
                    ffn_dim,
                    n_qubits_transformer,
                    n_qubits_ffn,
                    dropout,
                )
                for _ in range(num_blocks)
            ]
        )
        backend = qiskit.Aer.get_backend("aer_simulator")
        self.hybrid = Hybrid(n_qubits_ffn or n_qubits_transformer, backend, shots=100, shift=shift)
        self.classifier = nn.Linear(embed_dim, 1 if num_classes == 2 else num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_embedding(tokens)
        x = self.transformers(x)
        x = x.mean(dim=1)
        if self.classifier.out_features == 1:
            probs = self.hybrid(x)
            return torch.cat((probs, 1 - probs), dim=-1)
        else:
            logits = self.classifier(x)
            return F.softmax(logits, dim=-1)


def FCL(**kwargs):
    """Compatibility wrapper returning a HybridTransformerClassifier."""
    return HybridTransformerClassifier(**kwargs)


__all__ = [
    "QuantumCircuit",
    "HybridFunction",
    "Hybrid",
    "PositionalEncoder",
    "MultiHeadAttentionQuantum",
    "FeedForwardQuantum",
    "TransformerBlockQuantum",
    "HybridTransformerClassifier",
    "FCL",
]
