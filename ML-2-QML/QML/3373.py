"""
HybridTextClassifier – quantum‑enhanced implementation.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
import numpy as np
import qiskit
from qiskit.circuit.random import random_circuit


# --------------------------------------------------------------------------- #
#  Quantum convolutional front‑end
# --------------------------------------------------------------------------- #
class ConvQuantum:
    """Quanvolution filter implemented with Qiskit.

    The filter maps a 2‑D patch of input data to a vector of
    average |1> probabilities, one per qubit.  The returned vector
    can be interpreted as a feature map for the transformer.
    """

    class QuanvCircuit:
        def __init__(self, kernel_size: int, backend, shots: int, threshold: float):
            self.n_qubits = kernel_size ** 2
            self._circuit = qiskit.QuantumCircuit(self.n_qubits)
            self.theta = [
                qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)
            ]
            for i in range(self.n_qubits):
                self._circuit.rx(self.theta[i], i)
            self._circuit.barrier()
            self._circuit += random_circuit(self.n_qubits, 2)
            self._circuit.measure_all()

            self.backend = backend
            self.shots = shots
            self.threshold = threshold

        def run(self, data):
            """Run the circuit on a single 2‑D patch."""
            data = np.reshape(data, (1, self.n_qubits))

            param_binds = []
            for dat in data:
                bind = {}
                for i, val in enumerate(dat):
                    bind[self.theta[i]] = np.pi if val > self.threshold else 0
                param_binds.append(bind)

            job = qiskit.execute(
                self._circuit,
                self.backend,
                shots=self.shots,
                parameter_binds=param_binds,
            )
            result = job.result().get_counts(self._circuit)

            # compute per‑qubit average probability of |1>
            probs = np.zeros(self.n_qubits)
            for key, val in result.items():
                for q in range(self.n_qubits):
                    if key[q] == "1":
                        probs[q] += val
            probs /= self.shots
            return probs

    def __init__(self, kernel_size: int = 2, shots: int = 100, threshold: float = 127):
        backend = qiskit.Aer.get_backend("qasm_simulator")
        self.circuit = self.QuanvCircuit(kernel_size, backend, shots, threshold)

    def run(self, data: np.ndarray) -> np.ndarray:
        """
        Apply the quanvolution filter to a 2‑D input patch.

        Returns:
            np.ndarray: vector of average |1> probabilities per qubit.
        """
        return self.circuit.run(data)


# --------------------------------------------------------------------------- #
#  Quantum transformer sub‑modules
# --------------------------------------------------------------------------- #
class MultiHeadAttentionQuantum(nn.Module):
    """Multi‑head attention with quantum‑encoded projections."""

    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int = 8):
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [i], "func": "rx", "wires": [i]}
                    for i in range(n_wires)
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
            for wire in range(self.n_wires - 1):
                tqf.cnot(q_device, wires=[wire, wire + 1])
            tqf.cnot(q_device, wires=[self.n_wires - 1, 0])
            return self.measure(q_device)

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, q_device: Optional[tq.QuantumDevice] = None):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.q_layer = self.QLayer()
        self.q_device = q_device or tq.QuantumDevice(n_wires=self.q_layer.n_wires)
        self.combine_heads = nn.Linear(embed_dim, embed_dim)

    def _apply_quantum_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        projections = []
        for token in x.unbind(dim=1):
            token = token.view(token.size(0), self.num_heads, -1)
            head_outputs = []
            for head in token.unbind(dim=1):
                qdev = self.q_device.copy(bsz=head.size(0), device=head.device)
                head_outputs.append(self.q_layer(head, qdev))
            projections.append(torch.stack(head_outputs, dim=1))
        return torch.stack(projections, dim=1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = x.size(0)
        k = self._apply_quantum_heads(x)
        q = self._apply_quantum_heads(x)
        v = self._apply_quantum_heads(x)
        # attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        out = torch.matmul(scores, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        return self.combine_heads(out)


class FeedForwardQuantum(nn.Module):
    """Quantum feed‑forward network."""

    class QLayer(tq.QuantumModule):
        def __init__(self, n_qubits: int):
            super().__init__()
            self.n_wires = n_qubits
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [i], "func": "rx", "wires": [i]}
                    for i in range(n_qubits)
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

    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int, dropout: float = 0.1):
        super().__init__()
        self.q_layer = self.QLayer(n_qubits)
        self.q_device = tq.QuantumDevice(n_wires=n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for token in x.unbind(dim=1):
            qdev = self.q_device.copy(bsz=token.size(0), device=token.device)
            outputs.append(self.q_layer(token, qdev))
        out = torch.stack(outputs, dim=1)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))


class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding (identical to classical)."""

    def __init__(self, embed_dim: int, max_len: int = 5000):
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


class TransformerBlockQuantum(nn.Module):
    """Single transformer block with quantum attention and feed‑forward."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        n_qubits_transformer: int,
        n_qubits_ffn: int,
        dropout: float = 0.1,
        q_device: Optional[tq.QuantumDevice] = None,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttentionQuantum(
            embed_dim, num_heads, dropout, q_device=q_device
        )
        if n_qubits_ffn > 0:
            self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits_ffn, dropout)
        else:
            self.ffn = nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out = self.attn(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


# --------------------------------------------------------------------------- #
#  Hybrid classifier
# --------------------------------------------------------------------------- #
class HybridTextClassifier(nn.Module):
    """Transformer‑based text classifier with a quantum quanvolution front‑end."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        conv_kernel: int = 2,
        conv_shots: int = 100,
        conv_threshold: float = 127,
        n_qubits_transformer: int = 8,
        n_qubits_ffn: int = 8,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv = ConvQuantum(conv_kernel, conv_shots, conv_threshold)
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.transformer = nn.Sequential(
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
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # token embedding
        tokens = self.token_embedding(x)  # (batch, seq_len, embed_dim)
        # extract patches for quanvolution
        batch, seq_len, embed_dim = tokens.shape
        # reshape to 2‑D patches of size (conv_kernel, conv_kernel)
        # Here we use a simple sliding window over the sequence dimension
        # to create overlapping patches.
        patches = []
        k = self.conv.circuit.n_qubits
        stride = int(math.sqrt(k))
        for i in range(0, seq_len - stride + 1, stride):
            patch = tokens[:, i : i + stride, :strides]  # (batch, stride, stride)
            patches.append(patch)
        patches = torch.stack(patches, dim=1)  # (batch, num_patches, stride, stride)
        # flatten patches for quantum circuit
        flat_patches = patches.reshape(-1, k)
        # run quanvolution
        conv_out = torch.tensor(
            np.stack([self.conv.run(p.cpu().numpy()) for p in flat_patches]),
            dtype=torch.float32,
            device=tokens.device,
        )
        # reshape back to sequence
        conv_out = conv_out.view(batch, -1, embed_dim)
        # positional encoding
        x = self.pos_encoder(conv_out)
        # transformer encoder
        x = self.transformer(x)
        # pool and classify
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)


__all__ = [
    "ConvQuantum",
    "MultiHeadAttentionQuantum",
    "FeedForwardQuantum",
    "PositionalEncoder",
    "TransformerBlockQuantum",
    "HybridTextClassifier",
]
