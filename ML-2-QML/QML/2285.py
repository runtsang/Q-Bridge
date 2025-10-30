"""Hybrid quantum‑classical transformer classifier for binary classification.

This module extends the pure‑classical implementation with quantum‑enhanced
components: a differentiable quantum expectation head and optional
quantum‑parameterised attention/FFN blocks.  The API is identical to the
classical version, allowing a user to toggle quantum functionality via
the `use_quantum` flag.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

import qiskit
from qiskit import assemble, transpile
import torchquantum as tq
import torchquantum.functional as tqf

__all__ = ["HybridQuantumTransformerClassifier"]


# --------------------------------------------------------------------------- #
#  Classical backbone (shared)
# --------------------------------------------------------------------------- #
class ConvFeatureExtractor(nn.Module):
    """Small CNN to extract image features – identical to the classical module."""
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 1),
            nn.Dropout2d(0.2),
            nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 1),
            nn.Dropout2d(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


# --------------------------------------------------------------------------- #
#  Positional encoding (shared)
# --------------------------------------------------------------------------- #
class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding – identical to the classical module."""
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                             (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


# --------------------------------------------------------------------------- #
#  Quantum expectation head
# --------------------------------------------------------------------------- #
class QuantumHybridFunction(torch.autograd.Function):
    """Autograd bridge between PyTorch and a Qiskit circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: "QuantumExpectationHead._Circuit", shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        result = circuit.run(inputs.tolist())
        return torch.tensor(result, dtype=inputs.dtype, device=inputs.device)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        shift = ctx.shift
        circuit = ctx.circuit
        grads = []
        for val, grad in zip(grad_output.tolist(), grad_output):
            right = circuit.run([val + shift])[0]
            left = circuit.run([val - shift])[0]
            grads.append(grad.item() * (right - left))
        return torch.tensor(grads, dtype=grad_output.dtype, device=grad_output.device), None, None


class QuantumExpectationHead(nn.Module):
    """Two‑qubit expectation head implemented with Aer."""
    class _Circuit:
        def __init__(self, n_qubits: int, backend, shots: int) -> None:
            self.circuit = qiskit.QuantumCircuit(n_qubits)
            self.n_qubits = n_qubits
            self.backend = backend
            self.shots = shots
            self.theta = qiskit.circuit.Parameter("theta")
            self.circuit.h(range(n_qubits))
            self.circuit.barrier()
            self.circuit.ry(self.theta, range(n_qubits))
            self.circuit.measure_all()

        def run(self, thetas: list[float]) -> list[float]:
            compiled = transpile(self.circuit, self.backend)
            qobj = assemble(
                compiled,
                shots=self.shots,
                parameter_binds=[{self.theta: theta} for theta in thetas],
            )
            job = self.backend.run(qobj)
            result = job.result()
            counts = result.get_counts()
            if isinstance(counts, list):
                return [self._expectation(c) for c in counts]
            return [self._expectation(counts)]

        def _expectation(self, count_dict: dict) -> float:
            states = np.array(list(count_dict.keys()), dtype=float)
            counts = np.array(list(count_dict.values()))
            probs = counts / self.shots
            return 0.5 * (states @ probs + 1)

    def __init__(self, n_qubits: int = 2, shots: int = 100, shift: float = np.pi / 2) -> None:
        super().__init__()
        self.backend = qiskit.Aer.get_backend("aer_simulator")
        self.circuit = self._Circuit(n_qubits, self.backend, shots)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is a 1‑D tensor of shape [batch]
        return QuantumHybridFunction.apply(x, self.circuit, self.shift)


# --------------------------------------------------------------------------- #
#  Quantum transformer blocks
# --------------------------------------------------------------------------- #
class MultiHeadAttentionQuantum(tq.QuantumModule):
    """Quantum‑parameterised attention head."""
    class _QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            self.parameters = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            for w, gate in enumerate(self.parameters):
                gate(q_device, wires=w)
            for w in range(self.n_wires - 1):
                tqf.cnot(q_device, wires=[w, w + 1])
            tqf.cnot(q_device, wires=[self.n_wires - 1, 0])
            return self.measure(q_device)

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1,
                 q_device: Optional[tq.QuantumDevice] = None) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.q_layer = self._QLayer(self.d_k)
        self.q_device = q_device or tq.QuantumDevice(n_wires=self.d_k)
        self.combine = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        heads = x.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        out_heads = []
        for h in heads.unbind(dim=1):
            qdev = self.q_device.copy(bsz=h.size(0), device=h.device)
            out_heads.append(self.q_layer(h, qdev))
        out = torch.stack(out_heads, dim=1).transpose(1, 2).contiguous().view(batch, seq, self.embed_dim)
        return self.combine(out)


class FeedForwardQuantum(tq.QuantumModule):
    """Quantum‑parameterised feed‑forward layer."""
    class _QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            self.parameters = nn.ModuleList([tq.RY(has_params=True, trainable=True) for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            for w, gate in enumerate(self.parameters):
                gate(q_device, wires=w)
            return self.measure(q_device)

    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.q_layer = self._QLayer(n_qubits)
        self.q_device = tq.QuantumDevice(n_wires=n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = []
        for token in x.unbind(dim=1):
            qdev = self.q_device.copy(bsz=token.size(0), device=token.device)
            outs.append(self.q_layer(token, qdev))
        out = torch.stack(outs, dim=0).permute(1, 0, 2)  # [batch, seq, n_qubits]
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))


class TransformerBlockQuantum(nn.Module):
    """Transformer block that uses quantum attention and FFN."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 n_qubits_attn: int, n_qubits_ffn: int,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits_ffn, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(src)
        src = self.norm1(src + self.dropout(attn_out))
        ffn_out = self.ffn(src)
        return self.norm2(src + self.dropout(ffn_out))


# --------------------------------------------------------------------------- #
#  Main hybrid classifier
# --------------------------------------------------------------------------- #
class HybridQuantumTransformerClassifier(nn.Module):
    """CNN + transformer + quantum or classical head for binary classification.

    Parameters
    ----------
    use_quantum : bool
        If True, the transformer blocks and the final head are quantum‑enhanced.
    quantum_ffn_qubits : int
        Number of qubits used in the quantum feed‑forward layers.
    """
    def __init__(self,
                 embed_dim: int = 128,
                 n_heads: int = 4,
                 n_blocks: int = 2,
                 d_ff: int = 256,
                 dropout: float = 0.1,
                 use_quantum: bool = False,
                 quantum_ffn_qubits: int = 4) -> None:
        super().__init__()
        self.backbone = ConvFeatureExtractor()
        self.flatten = nn.Flatten(start_dim=1)
        self.proj = nn.Linear(55815, embed_dim)
        self.pos_enc = PositionalEncoding(embed_dim)
        if use_quantum:
            self.transformer = nn.Sequential(
                *[TransformerBlockQuantum(embed_dim, n_heads, d_ff,
                                          n_qubits_attn=embed_dim // n_heads,
                                          n_qubits_ffn=quantum_ffn_qubits,
                                          dropout=dropout)
                  for _ in range(n_blocks)]
            )
            self.head_proj = nn.Linear(embed_dim, 1)
            self.head = QuantumExpectationHead()
        else:
            self.transformer = nn.Sequential(
                *[TransformerBlock(embed_dim, n_heads, d_ff, dropout)
                  for _ in range(n_blocks)]
            )
            self.head = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.flatten(x)
        x = self.proj(x).unsqueeze(1)
        x = self.pos_enc(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        if hasattr(self, "head_proj"):
            x = self.head_proj(x).squeeze(-1)
            logits = self.head(x)
        else:
            logits = self.head(x).squeeze(-1)
        probs = torch.sigmoid(logits)
        return torch.cat([probs, 1 - probs], dim=-1)
