"""QTransformerTorch__gen428: Quantum‑enhanced transformer with Qiskit wrappers."""

from __future__ import annotations

import math
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN as QiskitSamplerQNN
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import StatevectorSampler as Sampler, StatevectorEstimator as Estimator
from qiskit.quantum_info import SparsePauliOp


# --------------------------------------------------------------------------- #
#  Quantum building blocks (TorchQuantum)
# --------------------------------------------------------------------------- #
class MultiHeadAttentionQuantum(nn.Module):
    """Quantum‑augmented multi‑head attention."""

    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int = 8) -> None:
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

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.q_layer = self.QLayer()
        self.combine = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq_len, _ = x.size()
        q = torch.stack([self.q_layer(token, self.q_layer.n_wires) for token in x.unbind(dim=1)], dim=1)
        k = q
        v = q
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)
        return self.combine(out)


class FeedForwardQuantum(nn.Module):
    """Quantum feed‑forward network."""

    class QLayer(tq.QuantumModule):
        def __init__(self, n_qubits: int) -> None:
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

    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)
        self.q_layer = self.QLayer(n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.stack([self.q_layer(token, self.q_layer.n_qubits) for token in x.unbind(dim=1)], dim=1)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))


# --------------------------------------------------------------------------- #
#  Transformer block (quantum)
# --------------------------------------------------------------------------- #
class TransformerBlockQuantum(nn.Module):
    """Transformer block with quantum attention and optionally quantum FFN."""

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
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout)
        self.ffn = (
            FeedForwardQuantum(embed_dim, ffn_dim, n_qubits_ffn, dropout)
            if n_qubits_ffn > 0
            else FeedForwardClassical(embed_dim, ffn_dim, dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


# --------------------------------------------------------------------------- #
#  Positional encoder (same as classical)
# --------------------------------------------------------------------------- #
class PositionalEncoder(nn.Module):
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
#  Main quantum‑enhanced transformer
# --------------------------------------------------------------------------- #
class QTransformerTorch__gen428(nn.Module):
    """Hybrid transformer that optionally replaces classical sub‑modules with quantum ones."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        use_quantum_attention: bool = False,
        use_quantum_ffn: bool = False,
        n_qubits_transformer: int = 0,
        n_qubits_ffn: int = 0,
    ) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = PositionalEncoder(embed_dim)
        self.blocks = nn.Sequential(
            *[
                (
                    TransformerBlockQuantum(
                        embed_dim,
                        num_heads,
                        ffn_dim,
                        n_qubits_transformer if use_quantum_attention else 0,
                        n_qubits_ffn if use_quantum_ffn else 0,
                        dropout,
                    )
                    if use_quantum_attention or use_quantum_ffn
                    else TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout)
                )
                for _ in range(num_blocks)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        out_dim = num_classes if num_classes > 2 else 1
        self.classifier = nn.Linear(embed_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_emb(x)
        x = self.pos_emb(x)
        x = self.blocks(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)


# --------------------------------------------------------------------------- #
#  Quantum wrappers for sampler / estimator
# --------------------------------------------------------------------------- #
def SamplerQNN() -> QiskitSamplerQNN:
    """Return a Qiskit sampler QNN."""
    inputs = ParameterVector("input", 2)
    weights = ParameterVector("weight", 4)
    qc = QuantumCircuit(2)
    qc.ry(inputs[0], 0)
    qc.ry(inputs[1], 1)
    qc.cx(0, 1)
    qc.ry(weights[0], 0)
    qc.ry(weights[1], 1)
    qc.cx(0, 1)
    qc.ry(weights[2], 0)
    qc.ry(weights[3], 1)

    sampler = Sampler()
    return QiskitSamplerQNN(
        circuit=qc,
        input_params=inputs,
        weight_params=weights,
        sampler=sampler,
    )


def EstimatorQNN() -> QiskitEstimatorQNN:
    """Return a Qiskit estimator QNN."""
    inp = Parameter("input1")
    wgt = Parameter("weight1")
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.ry(inp, 0)
    qc.rx(wgt, 0)

    observable = SparsePauliOp.from_list([("Y" * qc.num_qubits, 1)])

    estimator = Estimator()
    return QiskitEstimatorQNN(
        circuit=qc,
        observables=observable,
        input_params=[inp],
        weight_params=[wgt],
        estimator=estimator,
    )


# --------------------------------------------------------------------------- #
#  Quantum fully‑connected layer
# --------------------------------------------------------------------------- #
def FCL() -> object:
    """Return a quantum fully‑connected layer implemented with Qiskit."""
    class QuantumFCL:
        def __init__(self, n_qubits: int = 1, shots: int = 100):
            self.backend = qiskit.Aer.get_backend("qasm_simulator")
            self.shots = shots
            self.circuit = QuantumCircuit(n_qubits)
            self.theta = Parameter("theta")
            self.circuit.h(range(n_qubits))
            self.circuit.ry(self.theta, range(n_qubits))
            self.circuit.measure_all()

        def run(self, thetas: Iterable[float]) -> np.ndarray:
            job = qiskit.execute(
                self.circuit,
                self.backend,
                shots=self.shots,
                parameter_binds=[{self.theta: theta} for theta in thetas],
            )
            result = job.result().get_counts(self.circuit)
            probs = np.array(list(result.values())) / self.shots
            states = np.array(list(result.keys())).astype(float)
            expectation = np.sum(states * probs)
            return np.array([expectation])

    return QuantumFCL()


__all__ = [
    "QTransformerTorch__gen428",
    "SamplerQNN",
    "EstimatorQNN",
    "FCL",
]
