"""Hybrid transformer classifier with quantum attention and feed‑forward modules.

The quantum implementation relies on TorchQuantum for efficient
simulation of the quantum sub‑modules and on Qiskit for an explicit
circuit execution.  The API mirrors the classical version so that the two
implementations can be swapped by simply importing from the
corresponding module.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer

# --------------------------------------------------------------------------- #
#  Quantum Self‑Attention (Qiskit) ------------------------------------------- #
# --------------------------------------------------------------------------- #

class QuantumSelfAttention:
    """Quantum self‑attention block implemented with Qiskit."""

    def __init__(self, n_qubits: int = 4, backend: Optional[qiskit.providers.BaseBackend] = None):
        self.n_qubits = n_qubits
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        circuit = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        circuit.measure(self.qr, self.cr)
        return circuit

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, inputs: torch.Tensor, shots: int = 1024) -> torch.Tensor:
        """
        Parameters
        ----------
        rotation_params : np.ndarray
            Flattened rotation angles of shape ``(3 * n_qubits,)``.
        entangle_params : np.ndarray
            Entangling angles of shape ``(n_qubits - 1,)``.
        inputs : torch.Tensor
            Input tensor ``(batch, seq_len, embed_dim)`` where
            ``embed_dim == n_qubits``.
        shots : int
            Number of shots for the simulation.

        Returns
        -------
        torch.Tensor
            Attention‑weighted representation of the same shape as ``inputs``.
        """
        batch, seq_len, embed_dim = inputs.shape
        if embed_dim!= self.n_qubits:
            raise ValueError("Input embedding dimension must match number of qubits.")
        probs = []
        for b in range(batch):
            probs_b = []
            for t in range(seq_len):
                rot = inputs[b, t].cpu().numpy()
                rot_pad = np.zeros(3 * self.n_qubits)
                rot_pad[: embed_dim] = rot
                circ = self._build_circuit(rot_pad, entangle_params)
                job = execute(circ, self.backend, shots=shots)
                counts = job.result().get_counts(circ)
                probs_vec = np.zeros(2 ** self.n_qubits)
                for state, cnt in counts.items():
                    probs_vec[int(state, 2)] = cnt / shots
                probs_b.append(probs_vec)
            probs.append(np.stack(probs_b, axis=0))
        probs = np.stack(probs, axis=0)  # (batch, seq_len, 2**n_qubits)
        # take first embed_dim components as a simple proxy for attention
        return torch.from_numpy(probs[..., :embed_dim]).float().to(inputs.device)


# --------------------------------------------------------------------------- #
#  Quantum Modules (TorchQuantum) ------------------------------------------ #
# --------------------------------------------------------------------------- #

class QuantumSelfAttentionLayer(tq.QuantumModule):
    """Quantum self‑attention layer that encodes each token into a set of
    rotation gates and applies a parameterised entanglement circuit.
    The output is the expectation value of Pauli‑Z on each qubit.
    """

    def __init__(self, n_qubits: int = 4):
        super().__init__()
        self.n_qubits = n_qubits
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [i], "func": "rx", "wires": [i]}
                for i in range(n_qubits)
            ]
        )
        self.parameters = nn.ModuleList(
            [tq.RX(has_params=True, trainable=True) for _ in range(n_qubits)]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
        # flatten to (batch, n_qubits)
        x = x.reshape(x.size(0), -1)
        if x.size(1) < self.n_qubits:
            padding = torch.zeros(x.size(0), self.n_qubits - x.size(1), device=x.device)
            x = torch.cat([x, padding], dim=1)
        elif x.size(1) > self.n_qubits:
            x = x[:, :self.n_qubits]
        self.encoder(q_device, x)
        for wire, gate in enumerate(self.parameters):
            gate(q_device, wires=wire)
        return self.measure(q_device)


class MultiHeadAttentionQuantum(nn.Module):
    """Multi‑head attention where each head is implemented by a
    QuantumSelfAttentionLayer.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        n_qubits_per_head: int = 4,
    ):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.q_layers = nn.ModuleList(
            [
                QuantumSelfAttentionLayer(n_qubits=n_qubits_per_head)
                for _ in range(num_heads)
            ]
        )
        self.combine = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.size()
        qdev = tq.QuantumDevice(n_wires=self.q_layers[0].n_qubits, bsz=batch)
        heads = x.view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        outputs = []
        for i, q_layer in enumerate(self.q_layers):
            head_out = q_layer(heads[:, i, :], qdev)
            outputs.append(head_out)
        out = torch.stack(outputs, dim=1).transpose(1, 2).contiguous()
        out = out.view(batch, seq_len, self.embed_dim)
        return self.combine(self.dropout(out))


class FeedForwardQuantum(nn.Module):
    """Feed‑forward network realised by a quantum module followed by a
    classical linear projection.
    """

    class QLayer(tq.QuantumModule):
        def __init__(self, n_qubits: int):
            super().__init__()
            self.n_qubits = n_qubits
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
        bsz, seq_len, _ = x.size()
        qdev = self.q_device.copy(bsz=bsz, device=x.device)
        out = self.q_layer(x, qdev)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))


class TransformerBlockQuantum(nn.Module):
    """Transformer block that uses a quantum attention sub‑module and a
    quantum feed‑forward network.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        n_qubits_per_head: int = 4,
        n_qubits_ffn: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.attn = MultiHeadAttentionQuantum(
            embed_dim, num_heads, dropout, n_qubits_per_head
        )
        self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits_ffn, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""

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


class HybridTransformerClassifier(nn.Module):
    """Quantum‑enabled transformer classifier that mirrors the classical
    :class:`HybridTransformerClassifier`.  The only differences are
    that the attention and feed‑forward layers are quantum
    implementations and an optional Qiskit‑based self‑attention
    circuit can be used as a drop‑in replacement.
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
        use_qiskit_self_attention: bool = False,
        n_qubits_per_head: int = 4,
        n_qubits_ffn: int = 4,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)
        if use_qiskit_self_attention:
            self.qiskit_sa = QuantumSelfAttention(n_qubits=embed_dim)
        else:
            blocks = [
                TransformerBlockQuantum(
                    embed_dim, num_heads, ffn_dim, n_qubits_per_head, n_qubits_ffn, dropout
                )
                for _ in range(num_blocks)
            ]
            self.transformers = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            LongTensor of token indices with shape ``(batch, seq_len)``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(batch, num_classes)`` or ``(batch, 1)`` for binary.
        """
        tokens = self.token_embedding(x)
        x = self.pos_embedding(tokens)
        if hasattr(self, "qiskit_sa"):
            rotation_params = np.random.rand(3 * self.qiskit_sa.n_qubits)
            entangle_params = np.random.rand(self.qiskit_sa.n_qubits - 1)
            x = self.qiskit_sa.run(rotation_params, entangle_params, x)
        else:
            x = self.transformers(x)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)


__all__ = [
    "QuantumSelfAttention",
    "QuantumSelfAttentionLayer",
    "MultiHeadAttentionQuantum",
    "FeedForwardQuantum",
    "TransformerBlockQuantum",
    "PositionalEncoder",
    "HybridTransformerClassifier",
]
