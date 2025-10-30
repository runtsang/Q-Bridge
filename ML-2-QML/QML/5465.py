import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.utils import algorithm_globals
from qiskit_machine_learning.optimizers import COBYLA
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple, List, Optional

# ----------------------------------------------------------------------
# Fraud‑detection style parameterisation (quantum placeholder)
# ----------------------------------------------------------------------
@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _apply_layer(modes: Sequence, params: FraudLayerParameters, *, clip: bool) -> None:
    # Simplified placeholder – in practice would use Strawberry Fields ops
    pass

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> QuantumCircuit:
    # Placeholder returning a simple circuit
    qc = QuantumCircuit(2)
    return qc

# ----------------------------------------------------------------------
# QCNN convolutional helpers
# ----------------------------------------------------------------------
def conv_circuit(params):
    target = QuantumCircuit(2)
    target.rz(-math.pi / 2, 1)
    target.cx(1, 0)
    target.rz(params[0], 0)
    target.ry(params[1], 1)
    target.cx(0, 1)
    target.ry(params[2], 1)
    target.cx(1, 0)
    target.rz(math.pi / 2, 0)
    return target

def conv_layer(num_qubits, param_prefix):
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc = qc.compose(conv_circuit(params[param_index : (param_index + 3)]), [q1, q2])
        qc.barrier()
        param_index += 3
    qc_inst = qc.to_instruction()
    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, qubits)
    return qc

def pool_circuit(params):
    target = QuantumCircuit(2)
    target.rz(-math.pi / 2, 1)
    target.cx(1, 0)
    target.rz(params[0], 0)
    target.ry(params[1], 1)
    target.cx(0, 1)
    target.ry(params[2], 1)
    return target

def pool_layer(sources, sinks, param_prefix):
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for source, sink in zip(sources, sinks):
        qc = qc.compose(pool_circuit(params[param_index : (param_index + 3)]), [source, sink])
        qc.barrier()
        param_index += 3
    qc_inst = qc.to_instruction()
    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, range(num_qubits))
    return qc

# ----------------------------------------------------------------------
# Quantum‑enhanced transformer components
# ----------------------------------------------------------------------
class MultiHeadAttentionBase(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError

class MultiHeadAttentionQuantum(MultiHeadAttentionBase):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, use_qcnn: bool = False) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.use_qcnn = use_qcnn
        if use_qcnn:
            # QCNN convolutional layer to generate attention weights
            self.qc = conv_layer(num_heads, "θ")
        else:
            # Simple linear mapping as placeholder
            self.linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.use_qcnn:
            # Placeholder: return zeros of appropriate shape
            batch_size, seq_len, embed_dim = x.size()
            return torch.zeros_like(x)
        else:
            return self.linear(x)

class FeedForwardBase(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class FeedForwardQuantum(FeedForwardBase):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1, use_qcnn: bool = False) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.use_qcnn = use_qcnn
        if use_qcnn:
            # QCNN convolutional layer as feed‑forward
            self.conv = conv_layer(ffn_dim, "θ")
        else:
            self.linear1 = nn.Linear(embed_dim, ffn_dim)
            self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_qcnn:
            # Placeholder: return input unchanged
            return x
        else:
            return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerBlockBase(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class TransformerBlockQuantum(TransformerBlockBase):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1, use_qcnn: bool = False) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout, use_qcnn=use_qcnn)
        self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, dropout, use_qcnn=use_qcnn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class PositionalEncoder(nn.Module):
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

# ----------------------------------------------------------------------
# Unified transformer (quantum variant)
# ----------------------------------------------------------------------
class QTransformerUnified(nn.Module):
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
        n_qlayers: int = 1,
        use_qcnn: bool = False,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)
        blocks = []
        for _ in range(num_blocks):
            block = TransformerBlockQuantum(
                embed_dim,
                num_heads,
                ffn_dim,
                dropout,
                use_qcnn=use_qcnn,
            )
            blocks.append(block)
        self.transformers = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)
        # Metadata
        self.encoding = list(range(vocab_size))
        self.weight_sizes = []
        for block in self.transformers:
            for layer in [block.attn, block.ffn]:
                if hasattr(layer, "weight"):
                    w = layer.weight.numel()
                    b = layer.bias.numel() if hasattr(layer, "bias") else 0
                    self.weight_sizes.append(w + b)
        self.observables = [SparsePauliOp.from_list([("Z" + "I" * (embed_dim - 1), 1)])]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_embedding(tokens)
        x = self.transformers(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)

    def get_metadata(self) -> Tuple[List[int], List[int], List[SparsePauliOp]]:
        return self.encoding, self.weight_sizes, self.observables

__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "conv_layer",
    "pool_layer",
    "MultiHeadAttentionBase",
    "MultiHeadAttentionQuantum",
    "FeedForwardBase",
    "FeedForwardQuantum",
    "TransformerBlockBase",
    "TransformerBlockQuantum",
    "PositionalEncoder",
    "QTransformerUnified",
]
