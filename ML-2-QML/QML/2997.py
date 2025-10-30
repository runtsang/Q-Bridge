import math
from typing import Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class QAttention(tq.QuantumModule):
    """Quantum‑augmented multi‑head attention."""

    class Encoder(tq.QuantumModule):
        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.encoders = [
                tq.GeneralEncoder(
                    [
                        {
                            "input_idx": [i],
                            "func": "rx",
                            "wires": [i],
                        }
                        for i in range(n_wires)
                    ]
                )
            ]
            self.params = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoders[0](q_device, x)
            for wire, gate in enumerate(self.params):
                gate(q_device, wires=wire)
            return self.measure(q_device)

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        q_device: tq.QuantumDevice | None = None,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.q_device = q_device or tq.QuantumDevice(
            n_wires=8 * num_heads, device="cpu"
        )
        self.encoder = self.Encoder(self.q_device.n_wires)
        self.proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

    def _apply_quantum(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        outputs = []
        for token in x.unbind(dim=1):
            qdev = self.q_device.copy(bsz=token.size(0), device=token.device)
            outputs.append(self.encoder(token, qdev))
        return torch.stack(outputs, dim=1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        proj = self.proj(x)
        quantum_out = self._apply_quantum(proj)
        attn_scores = torch.matmul(quantum_out, quantum_out.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_scores = F.softmax(attn_scores, dim=-1)
        attn_scores = self.dropout(attn_scores)
        return torch.matmul(attn_scores, quantum_out)


class QFeedForward(tq.QuantumModule):
    """Feed‑forward network realised by a quantum circuit."""

    class Encoder(tq.QuantumModule):
        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.encoders = [
                tq.GeneralEncoder(
                    [
                        {
                            "input_idx": [i],
                            "func": "ry",
                            "wires": [i],
                        }
                        for i in range(n_wires)
                    ]
                )
            ]
            self.params = nn.ModuleList(
                [tq.RY(has_params=True, trainable=True) for _ in range(n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoders[0](q_device, x)
            for wire, gate in enumerate(self.params):
                gate(q_device, wires=wire)
            return self.measure(q_device)

    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        n_qubits: int = 8,
        dropout: float = 0.1,
        q_device: tq.QuantumDevice | None = None,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.n_qubits = n_qubits
        self.dropout = nn.Dropout(dropout)
        self.q_device = q_device or tq.QuantumDevice(n_wires=n_qubits)
        self.encoder = self.Encoder(n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        outputs = []
        for token in x.unbind(dim=1):
            qdev = self.q_device.copy(bsz=token.size(0), device=token.device)
            outputs.append(self.encoder(token, qdev))
        out = torch.stack(outputs, dim=1)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))


class TransformerBlockQuantum(nn.Module):
    """Transformer block that uses QAttention and QFeedForward."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        n_qubits_transformer: int,
        n_qubits_ffn: int,
        dropout: float = 0.1,
        q_device: tq.QuantumDevice | None = None,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = QAttention(embed_dim, num_heads, dropout, q_device)
        self.ffn = QFeedForward(embed_dim, ffn_dim, n_qubits_ffn, dropout, q_device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        return self.norm2(x + ffn_out)


class HybridTransformerClassifier(nn.Module):
    """
    Hybrid transformer classifier that can operate in a fully quantum mode
    or fall back to a classical backbone for compatibility.

    Parameters
    ----------
    vocab_size: int
        Size of the token vocabulary.
    embed_dim: int
        Dimensionality of token embeddings.
    num_heads: int
        Number of attention heads.
    num_blocks: int
        Number of transformer blocks.
    ffn_dim: int
        Hidden size of the feed‑forward sub‑network.
    num_classes: int
        Number of target classes.
    dropout: float
        Drop‑out probability.
    n_qubits_transformer: int
        Number of qubits per transformer block (0 → use classical).
    n_qubits_ffn: int
        Number of qubits per feed‑forward block (0 → use classical).
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
        n_qubits_transformer: int = 0,
        n_qubits_ffn: int = 0,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)

        if n_qubits_transformer > 0 and n_qubits_ffn > 0:
            blocks = [
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
        else:
            blocks = [
                TransformerBlock(
                    embed_dim, num_heads, ffn_dim, dropout
                )
                for _ in range(num_blocks)
            ]

        self.blocks = nn.ModuleList(blocks)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(x)
        x = self.pos_encoder(x)
        for blk in self.blocks:
            x = blk(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)

    def get_weight_sizes(self) -> List[int]:
        """Return a list of parameter counts for each linear layer."""
        sizes: List[int] = []
        for m in self.modules():
            if isinstance(m, nn.Linear):
                sizes.append(m.weight.numel() + m.bias.numel())
        return sizes


def build_quantum_classifier_circuit(
    num_qubits: int,
    depth: int,
) -> Tuple[QuantumCircuit, List[ParameterVector], List[ParameterVector], List[SparsePauliOp]]:
    """Construct a Qiskit circuit for a variational classifier.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    depth : int
        Number of variational layers.

    Returns
    -------
    circuit : QuantumCircuit
        The assembled quantum circuit.
    encodings : List[ParameterVector]
        Parameter vectors used for data encoding.
    weights : List[ParameterVector]
        Parameter vectors for variational parameters.
    observables : List[SparsePauliOp]
        Observable operators for measurement.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    for idx, qubit in enumerate(range(num_qubits)):
        circuit.rx(encoding[idx], qubit)

    w_idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[w_idx], qubit)
            w_idx += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]

    return circuit, [encoding], [weights], observables


__all__ = [
    "HybridTransformerClassifier",
    "build_quantum_classifier_circuit",
]
