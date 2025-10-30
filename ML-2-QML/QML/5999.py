import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchquantum as tq
import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate
from dataclasses import dataclass
from typing import Iterable, Sequence

@dataclass
class FraudLayerParametersQML:
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def build_fraud_detection_program(input_params: FraudLayerParametersQML,
                                  layers: Iterable[FraudLayerParametersQML]) -> sf.Program:
    program = sf.Program(2)
    with program.context as q:
        _apply_layer(q, input_params, clip=False)
        for layer in layers:
            _apply_layer(q, layer, clip=True)
    return program

def _apply_layer(modes: Sequence, params: FraudLayerParametersQML, clip: bool) -> None:
    BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
    for i, phase in enumerate(params.phases):
        Rgate(phase) | modes[i]
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        Sgate(r if not clip else _clip(r, 5), phi) | modes[i]
    BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
    for i, phase in enumerate(params.phases):
        Rgate(phase) | modes[i]
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        Dgate(r if not clip else _clip(r, 5), phi) | modes[i]
    for i, k in enumerate(params.kerr):
        Kgate(k if not clip else _clip(k, 1)) | modes[i]

class MultiHeadAttentionQuantum(tq.QuantumModule):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1,
                 q_device: tq.QuantumDevice | None = None) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.q_device = q_device or tq.QuantumDevice(n_wires=8)
        self.rx_gates = nn.ModuleList([tq.RX(has_params=False, trainable=False) for _ in range(self.q_device.n_wires)])
        self.cnot_gates = nn.ModuleList([tq.CNOT(has_params=False, trainable=False) for _ in range(self.q_device.n_wires - 1)])
        self.cnot_last = tq.CNOT(has_params=False, trainable=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, seq, dim = x.shape
        if dim!= self.embed_dim:
            raise ValueError("Input dim mismatch")
        out = []
        for token in x.unbind(dim=1):
            qdev = self.q_device.copy(bsz=token.size(0), device=token.device)
            for i, val in enumerate(token.unbind(dim=0)):
                self.rx_gates[i % qdev.n_wires](qdev, wires=[i % qdev.n_wires], theta=val)
            for i in range(qdev.n_wires - 1):
                self.cnot_gates[i](qdev, wires=[i, i + 1])
            self.cnot_last(qdev, wires=[qdev.n_wires - 1, 0])
            measure = tq.MeasureAll(tq.PauliZ)
            meas = measure(qdev)
            out.append(torch.tensor(meas, device=x.device, dtype=torch.float32))
        out = torch.stack(out, dim=1)
        return out

class FeedForwardQuantum(tq.QuantumModule):
    def __init__(self, n_qubits: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.dropout = nn.Dropout(dropout)
        self.q_device = tq.QuantumDevice(n_wires=n_qubits)
        self.linear1 = nn.Linear(n_qubits, n_qubits * 2)
        self.linear2 = nn.Linear(n_qubits * 2, n_qubits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, dim = x.shape
        out = []
        for token in x.unbind(dim=1):
            qdev = self.q_device.copy(bsz=token.size(0), device=token.device)
            for i, val in enumerate(token.unbind(dim=0)):
                tq.RX(has_params=False, trainable=False)(qdev, wires=[i % qdev.n_wires], theta=val)
            for i in range(qdev.n_wires - 1):
                tq.CNOT(has_params=False, trainable=False)(qdev, wires=[i, i + 1])
            tq.CNOT(has_params=False, trainable=False)(qdev, wires=[qdev.n_wires - 1, 0])
            measure = tq.MeasureAll(tq.PauliZ)
            meas = measure(qdev)
            out.append(torch.tensor(meas, device=x.device, dtype=torch.float32))
        out = torch.stack(out, dim=1)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))

class TransformerBlockQuantum(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1,
                 n_qubits: int = 8) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardQuantum(n_qubits, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.attn.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.ffn.dropout(ffn_out))

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
        return x + self.pe[:, :x.size(1)]

class QTransformerTorch(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 n_qubits: int,
                 dropout: float = 0.1,
                 num_classes: int = 2) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.transformer = nn.Sequential(*[TransformerBlockQuantum(embed_dim, num_heads, dropout, n_qubits)
                                          for _ in range(num_blocks)])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_encoder(tokens)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)

__all__ = [
    "FraudLayerParametersQML",
    "build_fraud_detection_program",
    "MultiHeadAttentionQuantum",
    "FeedForwardQuantum",
    "TransformerBlockQuantum",
    "PositionalEncoder",
    "QTransformerTorch",
]
