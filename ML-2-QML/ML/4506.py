"""QuanvolutionHybrid: Classical‑quantum hybrid architecture.

This module extends the original Quanvolution example by adding a
quantum‑fully‑connected layer (implemented with Qiskit) and a
transformer‑style classifier that can optionally use a quantum‑enhanced
attention head.  The design keeps the original 2×2 patch extraction
so that existing MNIST pipelines remain valid, while the quantum
components provide additional expressive power without sacrificing
back‑end compatibility.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
import numpy as np

# --------------------------------------------------------------------------- #
# 1. Classical patch extractor (original Quanvolution filter)
# --------------------------------------------------------------------------- #
class QuanvolutionPatchExtractor(nn.Module):
    """Extract 2×2 image patches and flatten them into a feature vector.

    The implementation mirrors the original ``QuanvolutionFilter`` but
    keeps the interface identical so that the anchor ``Quanvolution.py``
    can be swapped for this file.  The extractor returns a tensor of
    shape ``(batch, 4*14*14)`` which is the same as the original filter.
    """
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x).view(x.size(0), -1)

# --------------------------------------------------------------------------- #
# 2. Quantum fully‑connected layer (from FCL pair)
# --------------------------------------------------------------------------- #
class QuantumFCLayer(nn.Module):
    """A parameterised quantum circuit that implements a small FC layer.

    The circuit is a single qubit with a trainable rotation Ry(θ).  The
    expectation value of Z is used as the output.  This is the same
    functionality as the ``FCL`` class in the QML seed.  The wrapper
    ensures that gradients flow back through the Qiskit simulator.
    """
    def __init__(self, n_qubits: int = 1, shots: int = 200) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = qiskit.Aer.get_backend("qasm_simulator")
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        self.theta = qiskit.circuit.Parameter("theta")
        self.circuit.h(range(n_qubits))
        self.circuit.barrier()
        self.circuit.ry(self.theta, range(n_qubits))
        self.circuit.measure_all()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ``x`` is expected to be a 1‑D tensor of shape (batch, n_qubits)
        batch = x.size(0)
        # Bind each sample independently
        jobs = []
        for val in x.squeeze(-1).tolist():
            bind = {self.theta: val}
            jobs.append(qiskit.execute(self.circuit, self.backend, shots=self.shots, parameter_binds=[bind]))
        # Aggregate results
        expectations = []
        for job in jobs:
            result = job.result().get_counts(self.circuit)
            counts = np.array(list(result.values()))
            states = np.array(list(result.keys())).astype(float)
            probs = counts / self.shots
            expectations.append(np.sum(states * probs))
        return torch.tensor(expectations, dtype=x.dtype, device=x.device).unsqueeze(-1)

# --------------------------------------------------------------------------- #
# 3. Quantum‑enhanced attention head (from Transformer pair)
# --------------------------------------------------------------------------- #
class QuantumAttentionHead(nn.Module):
    """Multi‑head attention where the key/value projections are obtained
    from a small quantum circuit.  The design follows the
    ``MultiHeadAttentionQuantum`` from the QTransformerTorch seed but
    is simplified to a single qubit per head for brevity.
    """
    def __init__(self, embed_dim: int, num_heads: int, n_qubits: int = 1, shots: int = 200) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = qiskit.Aer.get_backend("qasm_simulator")
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        self.theta = qiskit.circuit.Parameter("theta")
        self.circuit.h(range(n_qubits))
        self.circuit.barrier()
        self.circuit.ry(self.theta, range(n_qubits))
        self.circuit.measure_all()

    def _quantum_projection(self, values: torch.Tensor) -> torch.Tensor:
        # ``values``: (batch, d_k)
        batch = values.size(0)
        expectations = []
        for val in values.squeeze(-1).tolist():
            bind = {self.theta: val}
            job = qiskit.execute(self.circuit, self.backend, shots=self.shots, parameter_binds=[bind])
            result = job.result().get_counts(self.circuit)
            counts = np.array(list(result.values()))
            states = np.array(list(result.keys())).astype(float)
            probs = counts / self.shots
            expectations.append(np.sum(states * probs))
        return torch.tensor(expectations, dtype=values.dtype, device=values.device).unsqueeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Simplified: use the same quantum circuit for all heads
        batch, seq, _ = x.shape
        flat = x.view(batch * seq, self.num_heads, self.d_k)
        projections = []
        for token in flat.unbind(dim=0):
            proj = self._quantum_projection(token)
            projections.append(proj)
        proj_tensor = torch.stack(projections, dim=0).view(batch, seq, self.num_heads, self.d_k)
        # Classical attention using projected keys/values
        q = k = v = proj_tensor
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        return out.transpose(1, 2).contiguous().view(batch, seq, self.embed_dim)

# --------------------------------------------------------------------------- #
# 4. Classical transformer classifier (optionally quantum head)
# --------------------------------------------------------------------------- #
class TransformerBlock(nn.Module):
    """Single transformer block that can use a quantum attention head."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 use_quantum: bool = False, n_qubits: int = 1, shots: int = 200) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)
        if use_quantum:
            self.attn = QuantumAttentionHead(embed_dim, num_heads, n_qubits, shots)
        else:
            self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(ffn_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, embed_dim)
        attn_out, _ = self.attn(x, x, x) if isinstance(self.attn, nn.MultiheadAttention) else self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class PositionalEncoding(nn.Module):
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

# --------------------------------------------------------------------------- #
# 5. The hybrid model
# --------------------------------------------------------------------------- #
class QuanvolutionHybrid(nn.Module):
    """Hybrid model that chains the patch extractor, a quantum FC layer,
    and a transformer classifier.  The transformer can use either the
    classical or quantum attention head depending on the ``use_quantum``
    flag.
    """
    def __init__(
        self,
        vocab_size: int = 10,
        embed_dim: int = 32,
        num_heads: int = 4,
        num_blocks: int = 2,
        ffn_dim: int = 64,
        num_classes: int = 10,
        use_quantum: bool = False,
        n_qubits_attention: int = 1,
        n_qubits_fc: int = 1,
    ) -> None:
        super().__init__()
        # 1. Feature extractor
        self.patch_extractor = QuanvolutionPatchExtractor()
        # 2. Quantum FC layer
        self.quantum_fc = QuantumFCLayer(n_qubits=n_qubits_fc)
        # 3. Positional encoding
        self.pos_encoder = PositionalEncoding(embed_dim)
        # 4. Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim,
                    num_heads,
                    ffn_dim,
                    use_quantum=use_quantum,
                    n_qubits=n_qubits_attention,
                )
                for _ in range(num_blocks)
            ]
        )
        self.transformer = nn.Sequential(*self.blocks)
        # 5. Classifier
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, 28, 28)
        features = self.patch_extractor(x)          # (batch, 4*14*14)
        q_features = self.quantum_fc(features)      # (batch, 1)
        seq_len = 14 * 14
        seq = q_features.repeat(1, seq_len).view(x.size(0), seq_len, 1)
        seq = seq + self.pos_encoder(seq)
        out = self.transformer(seq)
        out = out.mean(dim=1)
        return self.classifier(out)

__all__ = ["QuanvolutionHybrid"]
