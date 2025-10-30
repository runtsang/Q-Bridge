"""Hybrid quantum regression model combining QCNN, quantum autoencoder, quantum transformer, and regression head."""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN, SamplerQNN
from qiskit_machine_learning.utils import algorithm_globals
from qiskit.quantum_info import SparsePauliOp

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    omega_0 = np.zeros(2 ** num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_wires, dtype=complex)
    omega_1[-1] = 1.0
    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels

def QCNNQNN(num_qubits: int = 8) -> EstimatorQNN:
    algorithm_globals.random_seed = 12345
    estimator = Estimator()
    def conv_circuit(params):
        target = QuantumCircuit(2)
        target.rz(-np.pi / 2, 1)
        target.cx(1, 0)
        target.rz(params[0], 0)
        target.ry(params[1], 1)
        target.cx(0, 1)
        target.ry(params[2], 1)
        target.cx(1, 0)
        target.rz(np.pi / 2, 0)
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
        for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
            qc = qc.compose(conv_circuit(params[param_index : (param_index + 3)]), [q1, q2])
            qc.barrier()
            param_index += 3
        qc_inst = qc.to_instruction()
        qc = QuantumCircuit(num_qubits)
        qc.append(qc_inst, qubits)
        return qc
    def pool_circuit(params):
        target = QuantumCircuit(2)
        target.rz(-np.pi / 2, 1)
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
    feature_map = ZFeatureMap(num_qubits)
    ansatz = QuantumCircuit(num_qubits, name="Ansatz")
    ansatz.compose(conv_layer(num_qubits, "c1"), range(num_qubits), inplace=True)
    ansatz.compose(pool_layer(list(range(num_qubits//2)), list(range(num_qubits//2, num_qubits)), "p1"), range(num_qubits), inplace=True)
    ansatz.compose(conv_layer(num_qubits//2, "c2"), range(num_qubits//2, num_qubits), inplace=True)
    ansatz.compose(pool_layer(list(range(num_qubits//4)), list(range(num_qubits//4, num_qubits//2)), "p2"), range(num_qubits//2, num_qubits), inplace=True)
    ansatz.compose(conv_layer(num_qubits//4, "c3"), range(num_qubits//2, num_qubits), inplace=True)
    ansatz.compose(pool_layer([0], [1], "p3"), range(num_qubits//2, num_qubits), inplace=True)
    circuit = QuantumCircuit(num_qubits)
    circuit.compose(feature_map, range(num_qubits), inplace=True)
    circuit.compose(ansatz, range(num_qubits), inplace=True)
    observables = []
    for i in range(num_qubits):
        pauli_str = "I"*i + "Z" + "I"*(num_qubits-i-1)
        observables.append(SparsePauliOp.from_list([(pauli_str, 1)]))
    qnn = EstimatorQNN(circuit=circuit.decompose(),
                       observables=observables,
                       input_params=feature_map.parameters,
                       weight_params=ansatz.parameters,
                       estimator=estimator)
    return qnn

def QuantumAutoencoderQNN(num_wires: int = 3) -> SamplerQNN:
    algorithm_globals.random_seed = 42
    sampler = Estimator()
    def ansatz(params):
        thetas = params[0]
        phis = params[1]
        target = QuantumCircuit(num_wires)
        for i in range(num_wires):
            target.rx(thetas[i], i)
            target.ry(phis[i], i)
        return target
    params = ParameterVector("θ", length=2 * num_wires)
    circuit = ansatz(params)
    qnn = SamplerQNN(circuit=circuit,
                     input_params=[],
                     weight_params=params,
                     interpret=lambda x: x,
                     output_shape=2 ** num_wires,
                     sampler=sampler)
    return qnn

class FeedForwardClassical(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))

class MultiHeadAttentionQuantum(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [idx], "func": "rx", "wires": [idx]} for idx in range(n_wires)]
            )
            self.parameters = nn.ModuleList([tq.RY(has_params=True, trainable=True) for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)
        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            for wire, gate in enumerate(self.parameters):
                gate(q_device, wires=wire)
            return self.measure(q_device)
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, q_device: tq.QuantumDevice | None = None):
        super().__init__()
        self.n_wires = 8
        self.q_layer = self.QLayer(self.n_wires)
        self.q_device = q_device
        self.combine_heads = nn.Linear(embed_dim, embed_dim)
    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        outputs = []
        for token in x.unbind(dim=1):
            qdev = self.q_device or tq.QuantumDevice(n_wires=self.n_wires, bsz=token.size(0), device=token.device)
            out = self.q_layer(token, qdev)
            outputs.append(out)
        out = torch.stack(outputs, dim=1)
        return self.combine_heads(out)

class FeedForwardQuantum(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self, n_qubits: int):
            super().__init__()
            self.n_wires = n_qubits
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [idx], "func": "rx", "wires": [idx]} for idx in range(n_qubits)]
            )
            self.parameters = nn.ModuleList([tq.RY(has_params=True, trainable=True) for _ in range(n_qubits)])
            self.measure = tq.MeasureAll(tq.PauliZ)
        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            for wire, gate in enumerate(self.parameters):
                gate(q_device, wires=wire)
            return self.measure(q_device)
    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int, dropout: float = 0.1):
        super().__init__()
        self.q_layer = self.QLayer(n_qubits)
        self.q_device = tq.QuantumDevice(n_wires=n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for token in x.unbind(dim=1):
            qdev = self.q_device.copy(bsz=token.size(0), device=token.device)
            outputs.append(self.q_layer(token, qdev))
        out = torch.stack(outputs, dim=1)
        out = self.linear1(self.dropout(out))
        return self.linear2(torch.relu(out))

class TransformerBlockQuantum(tq.QuantumModule):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, n_qubits_transformer: int, n_qubits_ffn: int, n_qlayers: int, q_device: tq.QuantumDevice | None = None, dropout: float = 0.1):
        super().__init__()
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout, q_device=q_device)
        if n_qubits_ffn > 0:
            self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits_ffn, dropout)
        else:
            self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class PositionalEncoder(tq.QuantumModule):
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class QuantumTransformerEncoder(tq.QuantumModule):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 ffn_dim: int,
                 num_blocks: int,
                 n_qubits_transformer: int,
                 n_qubits_ffn: int,
                 n_qlayers: int,
                 q_device: tq.QuantumDevice | None = None,
                 dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList([TransformerBlockQuantum(embed_dim, num_heads, ffn_dim, n_qubits_transformer, n_qubits_ffn, n_qlayers, q_device=q_device, dropout=dropout) for _ in range(num_blocks)])
        self.pos_encoder = PositionalEncoder(embed_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pos_encoder(x)
        for block in self.blocks:
            x = block(x)
        return x

class HybridRegressionQuantum(tq.QuantumModule):
    def __init__(self,
                 num_qubits: int = 8,
                 num_wires_autoencoder: int = 3,
                 embed_dim: int = 8,
                 num_heads: int = 4,
                 ffn_dim: int = 256,
                 num_blocks: int = 2,
                 n_qubits_transformer: int = 8,
                 n_qubits_ffn: int = 8,
                 n_qlayers: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        self.qcnn = QCNNQNN(num_qubits)
        self.autoencoder = QuantumAutoencoderQNN(num_wires_autoencoder)
        self.transformer = QuantumTransformerEncoder(embed_dim,
                                                     num_heads,
                                                     ffn_dim,
                                                     num_blocks,
                                                     n_qubits_transformer,
                                                     n_qubits_ffn,
                                                     n_qlayers,
                                                     dropout=dropout)
        self.head = nn.Linear(embed_dim, 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qc_features = self.qcnn(x)
        auto_output = self.autoencoder(qc_features)
        auto_output = auto_output.unsqueeze(1)
        transformer_output = self.transformer(auto_output)
        transformer_output = transformer_output.squeeze(1)
        return self.head(transformer_output).squeeze(-1)

__all__ = ["HybridRegressionQuantum"]
