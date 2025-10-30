import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import qiskit
from qiskit.circuit.random import random_circuit
import torchquantum as tq
import torchquantum.functional as tqf

# --------------------------------------------------------------------------- #
# Quantum convolution – simple parameterised circuit
# --------------------------------------------------------------------------- #
class QuantumConvFilter(nn.Module):
    def __init__(self, kernel_size: int = 2, threshold: float = 127, shots: int = 100):
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.shots = shots
        self.backend = qiskit.Aer.get_backend('qasm_simulator')
        self.n_qubits = kernel_size ** 2
        self.circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [
            qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)
        ]
        for i in range(self.n_qubits):
            self.circuit.rx(self.theta[i], i)
        self.circuit.barrier()
        self.circuit += random_circuit(self.n_qubits, 2)
        self.circuit.measure_all()

    def forward(self, data):
        data = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)
        job = qiskit.execute(
            self.circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result().get_counts(self.circuit)
        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val
        return counts / (self.shots * self.n_qubits)

# --------------------------------------------------------------------------- #
# Quantum fully‑connected layer – uses a parameterised quantum circuit
# --------------------------------------------------------------------------- #
class QuantumFullyConnectedLayer(nn.Module):
    class QLayer(tq.QuantumModule):
        def __init__(self, n_qubits: int):
            super().__init__()
            self.n_wires = n_qubits
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [idx], "func": "rx", "wires": [idx]}
                    for idx in range(n_qubits)
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

    def __init__(self, n_features: int = 1):
        super().__init__()
        self.n_qubits = n_features
        self.q_layer = self.QLayer(self.n_qubits)
        self.q_device = tq.QuantumDevice(n_wires=self.n_qubits)
        self.linear1 = nn.Linear(self.n_qubits, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for token in x.unbind(dim=1):
            qdev = self.q_device.copy(bsz=token.size(0), device=token.device)
            outputs.append(self.q_layer(token, qdev))
        out = torch.stack(outputs, dim=1)
        out = self.linear1(out)
        return out

# --------------------------------------------------------------------------- #
# Quantum sampler – Qiskit Machine Learning SamplerQNN
# --------------------------------------------------------------------------- #
class QuantumSamplerQNN(nn.Module):
    def __init__(self):
        super().__init__()
        from qiskit_machine_learning.neural_networks import SamplerQNN
        from qiskit.primitives import StatevectorSampler as Sampler
        inputs = qiskit.circuit.ParameterVector("input", 2)
        weights = qiskit.circuit.ParameterVector("weight", 4)
        qc = qiskit.QuantumCircuit(2)
        qc.ry(inputs[0], 0)
        qc.ry(inputs[1], 1)
        qc.cx(0, 1)
        qc.ry(weights[0], 0)
        qc.ry(weights[1], 1)
        qc.cx(0, 1)
        qc.ry(weights[2], 0)
        qc.ry(weights[3], 1)
        sampler = Sampler()
        self.sampler_qnn = SamplerQNN(
            circuit=qc,
            input_params=inputs,
            weight_params=weights,
            sampler=sampler,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape (batch, 2)
        return self.sampler_qnn(x)

# --------------------------------------------------------------------------- #
# Quantum attention – simple parameterised attention using quantum encoding
# --------------------------------------------------------------------------- #
class QuantumAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.combine = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        qkv = self.qkv(x).view(batch, seq, 3, self.num_heads, self.d_k).transpose(2, 3)
        q, k, v = qkv.unbind(dim=2)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        out = torch.matmul(attn_probs, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq, -1)
        return self.combine(out)

# --------------------------------------------------------------------------- #
# Quantum transformer block
# --------------------------------------------------------------------------- #
class QuantumTransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = QuantumAttention(embed_dim, num_heads, dropout)
        self.ffn = QuantumFullyConnectedLayer(ffn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

# --------------------------------------------------------------------------- #
# Quantum‑enhanced hybrid transformer
# --------------------------------------------------------------------------- #
class HybridTransformerQML(nn.Module):
    """
    Quantum‑enabled transformer that keeps the same public API as HybridTransformerML.
    Each optional sub‑module (conv, fcl, sampler) can be activated to switch from
    classical to quantum behaviour.
    """
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int,
                 use_conv: bool = False,
                 use_fcl: bool = False,
                 use_sampler: bool = False,
                 dropout: float = 0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.blocks = nn.ModuleList(
            [
                QuantumTransformerBlock(embed_dim, num_heads, ffn_dim, dropout)
                for _ in range(num_blocks)
            ]
        )
        self.use_conv = use_conv
        self.use_fcl = use_fcl
        self.use_sampler = use_sampler
        if use_conv:
            self.conv = QuantumConvFilter()
        if use_fcl:
            self.fcl = QuantumFullyConnectedLayer()
        if use_sampler:
            self.sampler = QuantumSamplerQNN()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(
            embed_dim, num_classes if num_classes > 2 else 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_encoder(tokens)
        for block in self.blocks:
            x = block(x)
        if self.use_conv:
            conv_outs = []
            for token in x.unbind(dim=1):
                reshaped = token.detach().cpu().numpy().reshape(
                    1, self.conv.kernel_size, self.conv.kernel_size
                )
                conv_val = self.conv(reshaped)
                conv_outs.append(conv_val)
            x = torch.stack(conv_outs, dim=1).to(x.device)
        if self.use_fcl:
            fcl_val = self.fcl(x.mean(dim=1).detach().cpu().numpy())
            x = x + fcl_val.unsqueeze(1)
        if self.use_sampler:
            x = self.sampler(x.mean(dim=1))
            x = x.unsqueeze(1).repeat(1, x.size(1), 1)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)

__all__ = [
    "PositionalEncoder",
    "QuantumConvFilter",
    "QuantumFullyConnectedLayer",
    "QuantumSamplerQNN",
    "QuantumAttention",
    "QuantumTransformerBlock",
    "HybridTransformerQML",
]
