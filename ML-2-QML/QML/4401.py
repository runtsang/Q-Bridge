"""Quantum‑enhanced transformer with TorchQuantum and Qiskit primitives."""

from __future__ import annotations

import math
from typing import Optional, Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer
from qiskit.quantum_info import Statevector

# ---------- Quantum attention ----------
class MultiHeadAttentionBase(tq.QuantumModule):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError

class MultiHeadAttentionQuantum(MultiHeadAttentionBase):
    """Attention that maps each head through a small quantum circuit."""
    class QLayer(tq.QuantumModule):
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
            for wire, gate in enumerate(self.parameters):
                gate(q_device, wires=wire)
            for i in range(self.n_wires - 1):
                tqf.cnot(q_device, wires=[i, i + 1])
            tqf.cnot(q_device, wires=[self.n_wires - 1, 0])
            return self.measure(q_device)

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1,
                 q_device: Optional[tq.QuantumDevice] = None) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.q_layer = self.QLayer(self.num_heads)
        self.q_device = q_device or tq.QuantumDevice(n_wires=self.num_heads)
        self.combine = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq, _ = x.size()
        proj = x.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        outputs = []
        for h in range(self.num_heads):
            head = proj[:, h, :, :].reshape(batch * seq, self.d_k)
            qdev = self.q_device.copy(bsz=batch * seq, device=head.device)
            out = self.q_layer(head, qdev).reshape(batch, seq, self.d_k)
            outputs.append(out)
        out = torch.stack(outputs, dim=1).transpose(1, 2).contiguous().view(batch, seq, self.embed_dim)
        return self.combine(out)

# ---------- Quantum feed‑forward ----------
class FeedForwardQuantum(tq.QuantumModule):
    """Feed‑forward realized by a quantum circuit followed by a classical linear map."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(n_wires)]
            )
            self.parameters = nn.ModuleList([tq.RY(has_params=True, trainable=True) for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            for wire, gate in enumerate(self.parameters):
                gate(q_device, wires=wire)
            return self.measure(q_device)

    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.q_layer = self.QLayer(n_qubits)
        self.q_device = tq.QuantumDevice(n_wires=n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim, bias=False)
        self.linear2 = nn.Linear(ffn_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.size()
        outputs = []
        for i in range(seq):
            token = x[:, i, :].reshape(batch, 1, -1)
            qdev = self.q_device.copy(bsz=batch, device=token.device)
            out = self.q_layer(token.squeeze(1), qdev).reshape(batch, self.n_qubits)
            outputs.append(out)
        out = torch.stack(outputs, dim=1)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))

# ---------- Quantum transformer block ----------
class TransformerBlockQuantum(tq.QuantumModule):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 ffn_dim: int,
                 n_qubits_transformer: int,
                 n_qubits_ffn: int,
                 dropout: float = 0.1,
                 q_device: Optional[tq.QuantumDevice] = None) -> None:
        super().__init__()
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout, q_device=q_device)
        self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits_ffn, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(q_device, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(q_device, x)
        return self.norm2(x + self.dropout(ffn_out))

# ---------- Positional encoding ----------
class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float32) *
                             (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

# ---------- Quantum kernel ----------
class QuantumKernel(tq.QuantumModule):
    """Fixed 4‑qubit Ry ansatz used as a kernel."""
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(self.n_wires)]
        )

    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(q_device, x)
        for i in range(self.n_wires):
            self.q_device.apply(tq.RY(-y[i], wires=i))
        return torch.abs(q_device.states.view(-1)[0])

class QuantumKernelAttention(MultiHeadAttentionBase):
    """Attention that uses a quantum kernel to compute similarity."""
    def __init__(self, embed_dim: int, num_heads: int, q_device: Optional[tq.QuantumDevice] = None,
                 dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.kernel = QuantumKernel()
        self.q_device = q_device or tq.QuantumDevice(n_wires=self.kernel.n_wires)

    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq, _ = x.size()
        out = []
        for b in range(batch):
            token = x[b]
            K = torch.zeros(seq, seq)
            for i in range(seq):
                for j in range(seq):
                    K[i, j] = self.kernel(self.q_device, token[i].unsqueeze(0), token[j].unsqueeze(0))
            attn = F.softmax(K, dim=-1)
            out.append(torch.matmul(attn, token))
        return torch.stack(out, dim=0)

# ---------- Quantum‑enhanced transformer ----------
class QTransformer(tq.QuantumModule):
    """
    Quantum‑enhanced transformer that replaces attention and feed‑forward layers
    with quantum circuits. The API mirrors the classical version for side‑by‑side
    experimentation.
    """
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int,
                 dropout: float = 0.1,
                 n_qubits_transformer: int = 0,
                 n_qubits_ffn: int = 0) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.q_device = tq.QuantumDevice(n_wires=max(n_qubits_transformer, n_qubits_ffn))
        blocks = []
        for _ in range(num_blocks):
            blocks.append(
                TransformerBlockQuantum(
                    embed_dim,
                    num_heads,
                    ffn_dim,
                    n_qubits_transformer,
                    n_qubits_ffn,
                    dropout=dropout,
                    q_device=self.q_device
                )
            )
        self.transformer = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(x)
        x = self.pos_encoder(x)
        for block in self.transformer:
            x = block(self.q_device, x)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)

# ---------- Fast estimator for quantum model ----------
class FastEstimatorQuantum:
    """Evaluator that runs a quantum transformer on a list of inputs and returns logits."""
    def __init__(self, model: tq.QuantumModule) -> None:
        self.model = model

    def evaluate(self, inputs: Iterable[torch.Tensor]) -> List[torch.Tensor]:
        self.model.eval()
        results = []
        with torch.no_grad():
            for inp in inputs:
                results.append(self.model(inp))
        return results

# ---------- Quantum self‑attention (Qiskit) ----------
class QuantumSelfAttention:
    """Quantum self‑attention block implemented with Qiskit."""
    def __init__(self, n_qubits: int = 4) -> None:
        self.n_qubits = n_qubits
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

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray,
            shots: int = 1024) -> dict:
        circuit = self._build_circuit(rotation_params, entangle_params)
        backend = Aer.get_backend("qasm_simulator")
        job = qiskit.execute(circuit, backend, shots=shots)
        return job.result().get_counts(circuit)

__all__ = [
    "MultiHeadAttentionBase",
    "MultiHeadAttentionQuantum",
    "FeedForwardQuantum",
    "TransformerBlockQuantum",
    "PositionalEncoder",
    "QuantumKernel",
    "QuantumKernelAttention",
    "QTransformer",
    "FastEstimatorQuantum",
    "QuantumSelfAttention",
]
