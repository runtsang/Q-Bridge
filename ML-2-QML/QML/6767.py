"""Hybrid transformer that can switch between classical and quantum sub‑modules.

The quantum path uses a quantum convolution (QuanvCircuit) as a feature
extractor followed by a transformer block that leverages quantum
multi‑head attention and a quantum feed‑forward network.  The
classical path mirrors the implementation in the ML module, ensuring
API compatibility.
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


class QuantumConv(nn.Module):
    """
    Quantum convolutional filter implemented with Qiskit.  It maps a
    ``kernel_size × kernel_size`` patch to a single scalar value
    (average probability of measuring |1>).
    """

    class QuanvCircuit:
        """Underlying Qiskit circuit that encodes the data into rotation angles."""

        def __init__(self, kernel_size: int, backend, shots: int, threshold: float):
            self.n_qubits = kernel_size ** 2
            self._circuit = qiskit.QuantumCircuit(self.n_qubits)
            self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
            for i in range(self.n_qubits):
                self._circuit.rx(self.theta[i], i)
            self._circuit.barrier()
            self._circuit += random_circuit(self.n_qubits, 2)
            self._circuit.measure_all()

            self.backend = backend
            self.shots = shots
            self.threshold = threshold

        def run(self, data):
            """Run the quantum circuit on classical data.

            Args:
                data: 2D array with shape (kernel_size, kernel_size).

            Returns:
                float: average probability of measuring |1> across qubits.
            """
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

            counts = 0
            for key, val in result.items():
                ones = sum(int(bit) for bit in key)
                counts += ones * val

            return counts / (self.shots * self.n_qubits)

    def __init__(self, kernel_size: int = 2, threshold: float = 127.0, shots: int = 100):
        super().__init__()
        backend = qiskit.Aer.get_backend("qasm_simulator")
        self.circuit = self.QuanvCircuit(kernel_size, backend, shots, threshold)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input embeddings of shape ``(B, seq_len, embed_dim)``.

        Returns
        -------
        torch.Tensor
            Convolved embeddings of shape ``(B, seq_len, embed_dim)``.
        """
        # Reshape to 4‑D: (B, C=1, H=seq_len, W=embed_dim)
        x = x.unsqueeze(1)
        # Convert to numpy for Qiskit
        x_np = x.detach().cpu().numpy()
        batch, _, seq_len, embed_dim = x_np.shape
        out = []
        for b in range(batch):
            seq_out = []
            for t in range(seq_len):
                patch = x_np[b, 0, t, :].reshape(int(math.sqrt(embed_dim)), -1)
                val = self.circuit.run(patch)
                seq_out.append(val)
            out.append(seq_out)
        out = torch.tensor(out, device=x.device).unsqueeze(-1)  # (B, seq_len, 1)
        return out


class MultiHeadAttentionQuantum(nn.Module):
    """
    Multi‑head attention where the linear projections are replaced with a
    simple quantum module that applies a parameterised rotation to each
    head.  The implementation follows the structure of the classical
    version but delegates the head projection to a quantum circuit.
    """

    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int = 8):
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
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

    def _apply_quantum(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        projections = []
        for token in x.unbind(dim=1):
            token = token.view(token.size(0), self.num_heads, -1)
            head_outputs = []
            for head in token.unbind(dim=1):
                qdev = self.q_device.copy(bsz=head.size(0), device=head.device)
                head_outputs.append(self.q_layer(head, qdev))
            projections.append(torch.stack(head_outputs, dim=1))
        return torch.stack(projections, dim=1).contiguous()

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = x.size(0)
        q = self._apply_quantum(x)
        k = self._apply_quantum(x)
        v = self._apply_quantum(x)
        # Standard scaled dot‑product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        out = torch.matmul(scores, v)
        return self.combine_heads(out.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim))


class FeedForwardQuantum(nn.Module):
    """
    Feed‑forward network realised by a quantum module followed by
    classical linear layers.
    """

    class QLayer(tq.QuantumModule):
        def __init__(self, n_qubits: int = 8):
            super().__init__()
            self.n_wires = n_qubits
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(n_qubits)]
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

    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int = 8, dropout: float = 0.1):
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


class TransformerBlockQuantum(nn.Module):
    """
    Transformer block that combines the quantum attention and feed‑forward
    modules.  If the feed‑forward part is not quantum (``n_qubits_ffn = 0``)
    it falls back to the classical feed‑forward network.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        n_qubits_ffn: int = 0,
        dropout: float = 0.1,
        q_device: Optional[tq.QuantumDevice] = None,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout, q_device)
        if n_qubits_ffn > 0:
            self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits_ffn, dropout)
        else:
            self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out = self.attn(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class PositionalEncoder(nn.Module):
    """
    Sinusoidal positional encoding identical to the classical version.
    """

    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class HybridTransformer(nn.Module):
    """
    Hybrid transformer that can operate in either classical or quantum mode.
    The quantum mode optionally uses quantum attention, a quantum feed‑forward
    network and a quantum convolutional feature extractor.
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
        quantum_conv: bool = False,
        n_qubits_ffn: int = 0,
        q_device: Optional[tq.QuantumDevice] = None,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)
        self.use_quantum = use_quantum
        self.quantum_conv = quantum_conv

        if quantum_conv:
            self.conv = QuantumConv()
            self.conv_proj = nn.Linear(1, embed_dim)

        if use_quantum:
            self.transformers = nn.Sequential(
                *[
                    TransformerBlockQuantum(
                        embed_dim,
                        num_heads,
                        ffn_dim,
                        n_qubits_ffn=n_qubits_ffn,
                        dropout=dropout,
                        q_device=q_device,
                    )
                    for _ in range(num_blocks)
                ]
            )
        else:
            self.transformers = nn.Sequential(
                *[
                    TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout)
                    for _ in range(num_blocks)
                ]
            )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input token indices of shape ``(B, seq_len)``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(B, num_classes)`` (or ``(B, 1)`` for binary).
        """
        tokens = self.token_embedding(x)
        x = self.pos_embedding(tokens)

        if self.quantum_conv:
            x = self.conv(x)          # (B, seq_len, 1)
            x = self.conv_proj(x)     # (B, seq_len, embed_dim)

        x = self.transformers(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)


__all__ = [
    "HybridTransformer",
    "QuantumConv",
    "MultiHeadAttentionQuantum",
    "FeedForwardQuantum",
    "TransformerBlockQuantum",
    "PositionalEncoder",
]
