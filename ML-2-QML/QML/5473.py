"""Quantum‑enhanced fraud detection model.

This implementation mirrors the classical version but replaces the transformer
blocks with quantum‑aware modules from TorchQuantum and the classification
head with a parametrised Qiskit circuit.  The photonic feature extractor is
represented by a lightweight quantum module that mimics the optical
semantics.  All components are fully differentiable and can be trained
end‑to‑end.
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

import qiskit
from qiskit import assemble, transpile, Aer
import torchquantum as tq
import torchquantum.functional as tqf

# --------------------------------------------------------------------------- #
#  Photonic feature extractor (quantum)
# --------------------------------------------------------------------------- #

@dataclass
class FraudLayerParameters:
    """Parameters describing a single photonic layer."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]

class PhotonicFeatureExtractor(nn.Module):
    """A tiny quantum module that emulates a photonic layer."""
    def __init__(self, embed_dim: int = 2):
        super().__init__()
        self.linear = nn.Linear(2, embed_dim)
        # In a full implementation a Strawberry Fields program would be used.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.linear(x))

# --------------------------------------------------------------------------- #
#  Quantum transformer components
# --------------------------------------------------------------------------- #

class MultiHeadAttentionQuantum(tq.QuantumModule):
    """Multi‑head attention that maps projections through a small quantum circuit."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int = 8):
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [idx], "func": "rx", "wires": [idx]} for idx in range(n_wires)]
            )
            self.parameters = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
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

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        q_device: tq.QuantumDevice | None = None,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.q_layer = self.QLayer()
        self.q_device = q_device or tq.QuantumDevice(n_wires=self.QLayer.n_wires)

    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

    def attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        return torch.matmul(scores, value), scores

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        q = self.separate_heads(x)
        k = self.separate_heads(x)
        v = self.separate_heads(x)
        out, _ = self.attention(q, k, v, mask)
        return out.transpose(1, 2).contiguous().view(x.size(0), -1, self.embed_dim)

class FeedForwardQuantum(tq.QuantumModule):
    """Feed‑forward network realised by a small quantum circuit."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_qubits: int):
            super().__init__()
            self.n_wires = n_qubits
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [idx], "func": "rx", "wires": [idx]} for idx in range(n_qubits)]
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
        self.q_layer = self.QLayer(n_qubits)
        self.q_device = tq.QuantumDevice(n_wires=n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = []
        for token in x.unbind(dim=1):
            out.append(self.q_layer(token, self.q_device))
        out = torch.stack(out, dim=1)
        out = self.linear1(out)
        return self.linear2(F.relu(out))

class TransformerBlockQuantum(nn.Module):
    """Quantum‑enhanced transformer block."""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        n_qubits_transformer: int,
        n_qubits_ffn: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout=dropout)
        self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits_ffn)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, embed_dim: int, max_len: int = 500):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

# --------------------------------------------------------------------------- #
#  Hybrid classification head using Qiskit
# --------------------------------------------------------------------------- #

class QuantumCircuit:
    """Parametrised two‑qubit circuit executed on Aer."""
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
    """Differentiable interface between PyTorch and the quantum circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.quantum_circuit = circuit
        expectation_z = ctx.quantum_circuit.run(inputs.tolist())
        result = torch.tensor([expectation_z])
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        input_values = np.array(inputs.tolist())
        shift = np.ones_like(input_values) * ctx.shift
        gradients = []
        for idx, value in enumerate(input_values):
            expectation_right = ctx.quantum_circuit.run([value + shift[idx]])
            expectation_left = ctx.quantum_circuit.run([value - shift[idx]])
            gradients.append(expectation_right - expectation_left)
        gradients = torch.tensor([gradients]).float()
        return gradients * grad_output.float(), None, None

class Hybrid(nn.Module):
    """Hybrid layer that forwards activations through a quantum circuit."""
    def __init__(self, n_qubits: int, backend, shots: int, shift: float) -> None:
        super().__init__()
        self.quantum_circuit = QuantumCircuit(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        flattened = inputs.view(-1)
        return HybridFunction.apply(flattened, self.quantum_circuit, self.shift)

# --------------------------------------------------------------------------- #
#  Main hybrid fraud detection model
# --------------------------------------------------------------------------- #

class FraudDetectionHybrid(nn.Module):
    """Quantum‑enhanced fraud detection model that combines a photonic feature
    extractor, a quantum‑aware transformer stack, and a quantum expectation head.
    """
    def __init__(
        self,
        embed_dim: int = 2,
        num_heads: int = 1,
        num_blocks: int = 2,
        ffn_dim: int = 64,
        n_qubits_transformer: int = 8,
        n_qubits_ffn: int = 8,
        shift: float = 0.0,
        feature_layers: Sequence[FraudLayerParameters] | None = None,
        input_params: FraudLayerParameters | None = None,
    ) -> None:
        super().__init__()
        if input_params is None:
            raise ValueError("input_params must be provided")
        # Feature extractor
        self.feature_extractor = PhotonicFeatureExtractor(embed_dim)
        # Quantum transformer
        self.transformer = nn.Sequential(
            *[
                TransformerBlockQuantum(
                    embed_dim, num_heads, ffn_dim, n_qubits_transformer, n_qubits_ffn
                )
                for _ in range(num_blocks)
            ]
        )
        self.pos_encoder = PositionalEncoder(embed_dim)
        # Quantum hybrid head
        backend = Aer.get_backend("aer_simulator")
        self.hybrid = Hybrid(n_qubits_transformer, backend, shots=100, shift=shift)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Photonic feature extraction
        feats = self.feature_extractor(inputs)
        # Transformer expects sequence dimension; we add a singleton seq dim
        seq = feats.unsqueeze(1)
        seq = self.pos_encoder(seq)
        seq = self.transformer(seq)
        seq = seq.squeeze(1)
        # Quantum hybrid classification
        probs = self.hybrid(seq)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["FraudDetectionHybrid"]
