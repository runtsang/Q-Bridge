import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


# ------------------------------------------------------------
# Quantum‑enhanced attention
# ------------------------------------------------------------
class MultiHeadAttentionQuantum(nn.Module):
    """Attention where each linear projection is applied through a small quantum circuit."""

    class _QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int) -> None:
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
            return self.measure(q_device)

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

        self.q_layer = self._QLayer(n_wires=self.d_k)
        self.q_device = tq.QuantumDevice(n_wires=self.d_k)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        # Split heads
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        # Apply quantum layer to each head
        q_heads = []
        for head in range(self.num_heads):
            head_x = x[:, head].contiguous().view(-1, self.d_k)
            qout = self.q_layer(head_x, self.q_device)
            q_heads.append(qout.view(batch_size, seq_len, self.d_k))
        q = torch.stack(q_heads, dim=1)  # (B, H, L, d_k)

        # Classical attention on quantum outputs
        scores = torch.matmul(q, q.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, q)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.out_proj(out)


# ------------------------------------------------------------
# Quantum‑enhanced feed‑forward
# ------------------------------------------------------------
class FeedForwardQuantum(nn.Module):
    """Feed‑forward layer that maps through a parameterised quantum circuit."""

    class _QLayer(tq.QuantumModule):
        def __init__(self, n_qubits: int) -> None:
            super().__init__()
            self.n_qubits = n_qubits
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [i], "func": "ry", "wires": [i]}
                    for i in range(n_qubits)
                ]
            )
            self.parameters = nn.ModuleList(
                [tq.RZ(has_params=True, trainable=True) for _ in range(n_qubits)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            for wire, gate in enumerate(self.parameters):
                gate(q_device, wires=wire)
            return self.measure(q_device)

    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.q_layer = self._QLayer(n_qubits)
        self.q_device = tq.QuantumDevice(n_wires=n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim, bias=True)
        self.linear2 = nn.Linear(ffn_dim, embed_dim, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply quantum circuit to each token
        batch_size, seq_len, _ = x.size()
        out = []
        for t in range(seq_len):
            token = x[:, t].contiguous().view(-1, self.q_device.n_wires)
            qout = self.q_layer(token, self.q_device)
            out.append(qout)
        qout = torch.stack(out, dim=1)
        qout = self.linear1(self.dropout(qout))
        return self.linear2(F.relu(qout))


# ------------------------------------------------------------
# Transformer block
# ------------------------------------------------------------
class TransformerBlockQuantum(nn.Module):
    """A transformer layer that combines quantum attention and feed‑forward."""

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

        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits_ffn, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(self.norm1(x))
        x = x + self.dropout(attn_out)
        ffn_out = self.ffn(self.norm2(x))
        return x + self.dropout(ffn_out)


# ------------------------------------------------------------
# Positional encoding
# ------------------------------------------------------------
class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding (identical to the classical version)."""

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


# ------------------------------------------------------------
# Optional quantum conv filter (using Qiskit)
# ------------------------------------------------------------
class QuantumConvFilter:
    """A very small Qiskit circuit that approximates a quanvolution filter."""

    def __init__(self, kernel_size: int = 2, shots: int = 100) -> None:
        import qiskit as qk
        from qiskit.circuit import Parameter
        from qiskit import QuantumCircuit, Aer, execute

        self.n_qubits = kernel_size ** 2
        self.backend = Aer.get_backend("qasm_simulator")
        self.shots = shots

        self.theta = [Parameter(f"theta{i}") for i in range(self.n_qubits)]
        self.circuit = QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            self.circuit.rx(self.theta[i], i)
        self.circuit.barrier()
        self.circuit += qk.circuit.random.random_circuit(self.n_qubits, 2)
        self.circuit.measure_all()

    def run(self, data: torch.Tensor) -> torch.Tensor:
        import numpy as np
        from qiskit import execute

        # Flatten data to 1‑D and bind parameters
        data_flat = data.reshape(1, self.n_qubits).cpu().numpy()
        binds = []
        for row in data_flat:
            bind = {self.theta[i]: np.pi if val > 0.5 else 0 for i, val in enumerate(row)}
            binds.append(bind)

        job = execute(
            self.circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=binds,
        )
        result = job.result().get_counts(self.circuit)

        # Compute average number of |1> outcomes
        total = 0
        for key, val in result.items():
            ones = sum(int(b) for b in key)
            total += ones * val
        return torch.tensor(total / (self.shots * self.n_qubits), dtype=torch.float32)


# ------------------------------------------------------------
# Optional quantum regressor (Qiskit EstimatorQNN)
# ------------------------------------------------------------
class QuantumEstimatorQNN(nn.Module):
    """A tiny quantum neural network that maps a 2‑D input to a scalar."""

    def __init__(self) -> None:
        super().__init__()
        from qiskit.circuit import Parameter
        from qiskit import QuantumCircuit
        from qiskit_machine_learning.neural_networks import EstimatorQNN
        from qiskit.primitives import StatevectorEstimator as Estimator

        input_param = Parameter("x")
        weight_param = Parameter("w")

        qc = QuantumCircuit(1)
        qc.h(0)
        qc.ry(input_param, 0)
        qc.rx(weight_param, 0)

        obs = [("Y", 1)]

        self.estimator_qnn = EstimatorQNN(
            circuit=qc,
            observables=obs,
            input_params=[input_param],
            weight_params=[weight_param],
            estimator=Estimator(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs shape: (B, 2) where first column is input, second is weight
        # For simplicity, we only use the first column as the quantum input
        return self.estimator_qnn(inputs[:, 0])


# ------------------------------------------------------------
# Hybrid transformer (quantum mode)
# ------------------------------------------------------------
class QTransformerHybrid(nn.Module):
    """Quantum‑enhanced transformer that can optionally include a quantum filter and regressor."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        n_qubits_attn: int = 8,
        n_qubits_ffn: int = 8,
        use_qconv: bool = False,
        use_qregressor: bool = False,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)

        self.blocks = nn.ModuleList(
            [
                TransformerBlockQuantum(
                    embed_dim,
                    num_heads,
                    ffn_dim,
                    n_qubits_attn,
                    n_qubits_ffn,
                    dropout,
                )
                for _ in range(num_blocks)
            ]
        )
        self.dropout = nn.Dropout(dropout)

        self.use_qconv = use_qconv
        if use_qconv:
            self.qconv = QuantumConvFilter()
        else:
            self.qconv = None

        self.use_qregressor = use_qregressor
        if use_qregressor:
            self.qregressor = QuantumEstimatorQNN()
        else:
            self.qregressor = None

        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: token indices, or a single‑channel image if ``use_qconv`` is True.
        """
        if self.use_qconv:
            # Input shape (B, 1, H, W)
            conv_out = self.qconv.run(x)
            # Convert to an index (simplified)
            idx = conv_out.squeeze().long()
            x = self.token_embedding(idx)
        else:
            x = self.token_embedding(x)

        x = self.pos_encoder(x)
        for blk in self.blocks:
            x = blk(x)

        pooled = self.dropout(x.mean(dim=1))

        if self.use_qregressor:
            return self.qregressor(pooled)
        return self.classifier(pooled)


__all__ = [
    "MultiHeadAttentionQuantum",
    "FeedForwardQuantum",
    "TransformerBlockQuantum",
    "PositionalEncoder",
    "QuantumConvFilter",
    "QuantumEstimatorQNN",
    "QTransformerHybrid",
]
