"""Hybrid transformer with quantum sub‑modules."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler as Sampler, StatevectorEstimator as Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit_machine_learning.neural_networks import SamplerQNN as QiskitSamplerQNN


class MultiHeadAttentionBase(nn.Module):
    """Shared logic for attention layers."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, use_bias: bool = False):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.attn_weights: torch.Tensor | None = None

    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        bs = x.size(0)
        return x.view(bs, -1, self.num_heads, self.d_k).transpose(1, 2)

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        return torch.matmul(scores, v), scores

    def downstream(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, batch_size: int, mask: torch.Tensor | None = None):
        q = self.separate_heads(q)
        k = self.separate_heads(k)
        v = self.separate_heads(v)
        out, self.attn_weights = self.attention(q, k, v, mask)
        return out.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)


class MultiHeadAttentionQuantum(MultiHeadAttentionBase):
    """Quantum‑based attention that applies a parameterised gate to each head."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int = 8):
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

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, q_device: tq.QuantumDevice | None = None):
        super().__init__(embed_dim, num_heads, dropout)
        self.q_layer = self.QLayer()
        self.q_device = q_device
        self.combine_heads = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        bs, seq_len, _ = x.size()
        if x.size(2)!= self.embed_dim:
            raise ValueError("input embedding size mismatch")
        k = self._apply_quantum_heads(x)
        q = self._apply_quantum_heads(x)
        v = self._apply_quantum_heads(x)
        out = self.downstream(q, k, v, bs, mask)
        return self.combine_heads(out)

    def _apply_quantum_heads(self, x: torch.Tensor) -> torch.Tensor:
        projections = []
        for token in x.unbind(dim=1):
            token = token.view(token.size(0), self.num_heads, -1)
            head_outputs = []
            for head in token.unbind(dim=1):
                qdev = self.q_device or tq.QuantumDevice(n_wires=self.q_layer.n_wires, bsz=head.size(0), device=head.device)
                head_outputs.append(self.q_layer(head, qdev))
            projections.append(torch.stack(head_outputs, dim=1))
        return torch.stack(projections, dim=1)


class FeedForwardBase(nn.Module):
    """Base feed‑forward network."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)


class FeedForwardQuantum(FeedForwardBase):
    """Quantum feed‑forward using a parameterised circuit."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_qubits: int):
            super().__init__()
            self.n_wires = n_qubits
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
            )
            self.parameters = nn.ModuleList([tq.RY(has_params=True, trainable=True) for _ in range(n_qubits)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            for w, gate in enumerate(self.parameters):
                gate(q_device, wires=w)
            return self.measure(q_device)

    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int, dropout: float = 0.1):
        super().__init__(embed_dim, ffn_dim, dropout)
        self.q_layer = self.QLayer(n_qubits)
        self.q_device = tq.QuantumDevice(n_wires=n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for token in x.unbind(dim=1):
            qdev = self.q_device.copy(bsz=token.size(0), device=token.device)
            outputs.append(self.q_layer(token, qdev))
        out = torch.stack(outputs, dim=1)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))


class TransformerBlockBase(nn.Module):
    """Base transformer block."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class TransformerBlockQuantum(TransformerBlockBase):
    """Quantum transformer block."""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        n_qubits_transformer: int,
        n_qubits_ffn: int,
        n_qlayers: int,
        q_device: tq.QuantumDevice | None = None,
        dropout: float = 0.1,
    ):
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout, q_device=q_device)
        if n_qubits_ffn > 0:
            self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits_ffn, dropout)
        else:
            self.ffn = FeedForwardBase(embed_dim, ffn_dim, dropout)

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
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


# Quantum auxiliary heads
class EstimatorQNN(nn.Module):
    """Quantum regression head."""
    def __init__(self) -> None:
        super().__init__()
        param = Parameter("w")
        qc = QuantumCircuit(1)
        qc.ry(param, 0)
        self.estimator = Estimator()
        self.qnn = QiskitEstimatorQNN(
            circuit=qc,
            observables=None,
            input_params=[],
            weight_params=[param],
            estimator=self.estimator,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Placeholder: return mean as regression output
        return x.mean(dim=1, keepdim=True)


class SamplerQNN(nn.Module):
    """Quantum sampling head."""
    def __init__(self) -> None:
        super().__init__()
        inputs = ParameterVector("x", 2)
        weights = ParameterVector("w", 4)
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
        self.qnn = QiskitSamplerQNN(circuit=qc, input_params=inputs, weight_params=weights, sampler=sampler)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(x, dim=-1)


class Autoencoder(nn.Module):
    """Quantum auto‑encoder based on a swap‑test."""
    def __init__(self) -> None:
        super().__init__()
        self.sampler = Sampler()
        num_latent, num_trash = 3, 2
        qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)
        ansatz = RealAmplitudes(num_latent + num_trash, reps=5)
        qc.compose(ansatz, range(0, num_latent + num_trash), inplace=True)
        qc.barrier()
        aux = num_latent + 2 * num_trash
        qc.h(aux)
        for i in range(num_trash):
            qc.cswap(aux, num_latent + i, num_latent + num_trash + i)
        qc.h(aux)
        qc.measure(aux, cr[0])
        self.circuit = qc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Placeholder: return input
        return x


class HybridTransformer(nn.Module):
    """Quantum‑enhanced transformer with optional classical heads."""
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        use_autoencoder: bool = False,
        use_sampler: bool = False,
        use_estimator: bool = False,
        q_device: tq.QuantumDevice | None = None,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)
        self.transformers = nn.Sequential(
            *[TransformerBlockQuantum(
                embed_dim, num_heads, ffn_dim,
                n_qubits_transformer=embed_dim,
                n_qubits_ffn=embed_dim,
                n_qlayers=1,
                q_device=q_device,
                dropout=dropout,
            ) for _ in range(num_blocks)]
        )
        self.dropout = nn.Dropout(dropout)
        if num_classes > 2:
            self.classifier = nn.Linear(embed_dim, num_classes)
        else:
            self.classifier = nn.Linear(embed_dim, 1)

        self.use_autoencoder = use_autoencoder
        if use_autoencoder:
            self.autoencoder = Autoencoder()

        self.use_sampler = use_sampler
        if use_sampler:
            self.sampler = SamplerQNN()

        self.use_estimator = use_estimator
        if use_estimator:
            self.estimator = EstimatorQNN()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_embedding(tokens)
        x = self.transformers(x)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor | None:
        if not self.use_autoencoder:
            return None
        return self.autoencoder(x)

    def sample(self, x: torch.Tensor) -> torch.Tensor | None:
        if not self.use_sampler:
            return None
        return self.sampler(x)

    def regress(self, x: torch.Tensor) -> torch.Tensor | None:
        if not self.use_estimator:
            return None
        return self.estimator(x)


__all__ = [
    "MultiHeadAttentionBase",
    "MultiHeadAttentionQuantum",
    "FeedForwardBase",
    "FeedForwardQuantum",
    "TransformerBlockBase",
    "TransformerBlockQuantum",
    "PositionalEncoder",
    "EstimatorQNN",
    "SamplerQNN",
    "Autoencoder",
    "HybridTransformer",
]
