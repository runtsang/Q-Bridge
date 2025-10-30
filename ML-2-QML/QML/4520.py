from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional

import torchquantum as tq
import torchquantum.functional as tqf
import qiskit
from qiskit import assemble, transpile

class QuanvolutionFilter(tq.QuantumModule):
    """Quantum quanvolution filter using a random two‑qubit kernel."""
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        # assume input shape (B,4,14,14) after classical conv
        patches = []
        for r in range(0, 14, 2):
            for c in range(0, 14, 2):
                data = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, data)
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)

class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding for transformer tokens."""
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

class MultiHeadAttentionQuantum(tq.QuantumModule):
    """Multi‑head attention that maps projections through a quantum module."""
    class QLayer(tq.QuantumModule):
        def __init__(self) -> None:
            super().__init__()
            self.n_wires = 8
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(self.n_wires)]
            )
            self.parameters = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(self.n_wires)]
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

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, q_device: Optional[tq.QuantumDevice] = None):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.q_layer = self.QLayer()
        self.q_device = q_device
        self.combine_heads = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.size()
        if embed_dim!= self.embed_dim:
            raise ValueError("Input embedding does not match layer embedding size")
        # split heads
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)  # (B, H, S, d_k)
        outputs = []
        for h in range(self.num_heads):
            head = x[:, h, :, :]  # (B, S, d_k)
            qdev = self.q_device or tq.QuantumDevice(self.q_layer.n_wires, bsz=head.size(0), device=head.device)
            out = self.q_layer(head, qdev)
            outputs.append(out)
        out = torch.stack(outputs, dim=1)  # (B, H, S, d_k)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.combine_heads(out)

class FeedForwardQuantum(tq.QuantumModule):
    """Feed‑forward network realised by a quantum module."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_qubits: int) -> None:
            super().__init__()
            self.n_wires = n_qubits
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [idx], "func": "rx", "wires": [idx]} for idx in range(n_qubits)]
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

    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int, dropout: float = 0.1):
        super().__init__()
        self.q_layer = self.QLayer(n_qubits)
        self.q_device = tq.QuantumDevice(n_qubits)
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
        return self.linear2(F.relu(out))

class TransformerBlockQuantum(tq.QuantumModule):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        n_qubits_transformer: int,
        n_qubits_ffn: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout)
        if n_qubits_ffn > 0:
            self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits_ffn, dropout)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(embed_dim, ffn_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(ffn_dim, embed_dim)
            )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class HybridFunction(torch.autograd.Function):
    """Differentiable interface between PyTorch and a quantum circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: 'QuantumCircuit', shift: float):
        ctx.shift = shift
        ctx.quantum_circuit = circuit
        expectation_z = ctx.quantum_circuit.run(inputs.tolist())
        result = torch.tensor([expectation_z])
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        input_values = np.array(inputs.tolist())
        shift = np.ones_like(input_values) * ctx.shift
        gradients = []
        for idx, value in enumerate(input_values):
            expectation_right = ctx.quantum_circuit.run([value + shift[idx]])
            expectation_left = ctx.quantum_circuit.run([value - shift[idx]])
            gradients.append(expectation_right - expectation_left)
        gradients = torch.tensor([gradients]).float()
        return gradients * grad_output.float(), None, None

class Hybrid(nn.Module):
    """Hybrid layer that forwards activations through a quantum circuit."""
    def __init__(self, n_qubits: int, backend, shots: int, shift: float):
        super().__init__()
        self.quantum_circuit = QuantumCircuit(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        squeezed = torch.squeeze(inputs) if inputs.shape!= torch.Size([1, 1]) else inputs[0]
        return HybridFunction.apply(squeezed, self.quantum_circuit, self.shift)

class QuantumCircuit:
    """Wrapper around a parametrised two‑qubit circuit executed on Aer."""
    def __init__(self, n_qubits: int, backend, shots: int):
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        all_qubits = list(range(n_qubits))
        self.theta = qiskit.circuit.Parameter("theta")
        self._circuit.h(all_qubits)
        self._circuit.barrier()
        self._circuit.ry(self.theta, all_qubits)
        self._circuit.measure_all()
        self.backend = backend
        self.shots = shots

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self._circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        def expectation(count_dict):
            counts = np.array(list(count_dict.values()))
            states = np.array(list(count_dict.keys())).astype(float)
            probabilities = counts / self.shots
            return np.sum(states * probabilities)
        if isinstance(result, list):
            return np.array([expectation(item) for item in result])
        return np.array([expectation(result)])

class EstimatorQNN(tq.QuantumModule):
    """Quantum feed‑forward regressor."""
    def __init__(self):
        super().__init__()
        params1 = [qiskit.circuit.Parameter("input1"), qiskit.circuit.Parameter("weight1")]
        qc1 = qiskit.QuantumCircuit(1)
        qc1.h(0)
        qc1.ry(params1[0], 0)
        qc1.rx(params1[1], 0)
        observable1 = qiskit.quantum_info.SparsePauliOp.from_list([("Y" * qc1.num_qubits, 1)])
        self.estimator = qiskit.primitives.StatevectorEstimator()
        self.estimator_qnn = qiskit_machine_learning.neural_networks.EstimatorQNN(
            circuit=qc1,
            observables=observable1,
            input_params=[params1[0]],
            weight_params=[params1[1]],
            estimator=self.estimator,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.estimator_qnn.predict(inputs)

class QuanvolutionNet(tq.QuantumModule):
    """
    Hybrid architecture that combines a classical quanvolution filter, a transformer encoder,
    and a flexible head that can perform classification, regression or a quantum‑style hybrid output.
    """
    def __init__(
        self,
        num_classes: int = 10,
        transformer_blocks: int = 2,
        embed_dim: int = 64,
        num_heads: int = 4,
        ffn_dim: int = 128,
        shift: float = 0.0,
        use_hybrid_head: bool = False,
        use_regressor: bool = False,
        n_qubits_transformer: int = 8,
        n_qubits_ffn: int = 8
    ):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)
        self.qfilter = QuanvolutionFilter()
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.transformers = nn.Sequential(
            *[TransformerBlockQuantum(embed_dim, num_heads, ffn_dim, n_qubits_transformer, n_qubits_ffn) for _ in range(transformer_blocks)]
        )
        self.dropout = nn.Dropout(0.1)
        self.use_hybrid_head = use_hybrid_head
        self.use_regressor = use_regressor
        self.proj = nn.Linear(4, embed_dim)
        if use_regressor:
            self.regressor_proj = nn.Linear(embed_dim, 2)
            self.head = EstimatorQNN()
        elif use_hybrid_head:
            self.head = Hybrid(embed_dim, qiskit.Aer.get_backend("aer_simulator"), shots=100, shift=shift)
        else:
            self.linear = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        patches = self.qfilter(x)
        patches = patches.permute(0, 2, 3, 1).reshape(x.size(0), -1, 4)
        x = self.proj(patches)
        x = self.pos_encoder(x)
        x = self.transformers(x)
        x = self.dropout(x.mean(dim=1))
        if self.use_regressor:
            x = self.regressor_proj(x)
            return self.head(x)
        elif self.use_hybrid_head:
            return self.head(x)
        else:
            logits = self.linear(x)
            return F.log_softmax(logits, dim=-1)

__all__ = [
    "QuanvolutionNet",
    "HybridFunction",
    "Hybrid",
    "EstimatorQNN",
    "QuanvolutionFilter",
    "TransformerBlockQuantum",
    "PositionalEncoder"
]
