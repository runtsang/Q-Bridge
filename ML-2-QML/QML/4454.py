import torch
import torch.nn as nn
import numpy as np
import torchquantum as tq
import torchquantum.functional as tqf
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from typing import Optional, Iterable, List
import torch.nn.functional as F

# ------------------------------------------------------------
# Quantum Self‑Attention (Qiskit implementation)
# ------------------------------------------------------------
class QuantumSelfAttention(nn.Module):
    def __init__(self, n_qubits: int):
        super().__init__()
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

    def forward(
        self,
        backend,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ) -> dict:
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = qiskit.execute(circuit, backend, shots=shots)
        return job.result().get_counts(circuit)

# ------------------------------------------------------------
# Quantum Sampler QNN (Qiskit implementation)
# ------------------------------------------------------------
class QuantumSamplerQNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs2 = ParameterVector("input", 2)
        self.weights2 = ParameterVector("weight", 4)
        qc2 = QuantumCircuit(2)
        qc2.ry(self.inputs2[0], 0)
        qc2.ry(self.inputs2[1], 1)
        qc2.cx(0, 1)
        qc2.ry(self.weights2[0], 0)
        qc2.ry(self.weights2[1], 1)
        qc2.cx(0, 1)
        qc2.ry(self.weights2[2], 0)
        qc2.ry(self.weights2[3], 1)
        self.circuit = qc2
        from qiskit_machine_learning.neural_networks import SamplerQNN
        from qiskit.primitives import StatevectorSampler as Sampler
        self.sampler = SamplerQNN(circuit=qc2, input_params=self.inputs2, weight_params=self.weights2, sampler=Sampler())

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Placeholder: return a fixed distribution
        return torch.tensor([0.5, 0.5], dtype=torch.float32)

# ------------------------------------------------------------
# Estimator utilities for quantum circuits
# ------------------------------------------------------------
class FastEstimatorQuantum:
    def __init__(self, circuit: QuantumCircuit):
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: List[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(self, observables: Iterable[BaseOperator], parameter_sets: List[List[float]]) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(observable) for observable in observables]
            results.append(row)
        return results

# ------------------------------------------------------------
# Quantum Transformer building blocks
# ------------------------------------------------------------
class MultiHeadAttentionQuantum(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, q_device: Optional[tq.QuantumDevice] = None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.combine_heads = nn.Linear(embed_dim, embed_dim)
        self.q_device = q_device or tq.QuantumDevice(n_wires=8)
        self.qlayer = self._create_quantum_layer()

    def _create_quantum_layer(self):
        class QLayer(tq.QuantumModule):
            def __init__(self):
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
        return QLayer()

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        k = self.k_linear(x)
        q = self.q_linear(x)
        v = self.v_linear(x)
        # Apply quantum layer to each token vector
        out = []
        for token in q.unbind(dim=1):
            qdev = self.q_device.copy(bsz=token.size(0), device=token.device)
            out.append(self.qlayer(token, qdev))
        out = torch.stack(out, dim=1)
        out = self.combine_heads(out)
        return out

class FeedForwardQuantum(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int, dropout: float = 0.1):
        super().__init__()
        self.n_qubits = n_qubits
        self.q_layer = self._create_quantum_layer()
        self.q_device = tq.QuantumDevice(n_wires=n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def _create_quantum_layer(self):
        class QLayer(tq.QuantumModule):
            def __init__(self, n_qubits):
                super().__init__()
                self.n_wires = n_qubits
                self.encoder = tq.GeneralEncoder(
                    [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
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
        return QLayer(self.n_qubits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for token in x.unbind(dim=1):
            qdev = self.q_device.copy(bsz=token.size(0), device=token.device)
            outputs.append(self.q_layer(token, qdev))
        out = torch.stack(outputs, dim=1)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))

class TransformerBlockQuantum(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, n_qubits_transformer: int, n_qubits_ffn: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout)
        if n_qubits_ffn > 0:
            self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits_ffn, dropout)
        else:
            self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

# ------------------------------------------------------------
# Classical Feed‑Forward (needed for hybrid blocks)
# ------------------------------------------------------------
class FeedForwardClassical(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

# ------------------------------------------------------------
# Positional Encoding (shared)
# ------------------------------------------------------------
class PositionalEncoder(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-np.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]

# ------------------------------------------------------------
# Quantum Text Classifier (with optional sampler head)
# ------------------------------------------------------------
class TextClassifierQuantum(nn.Module):
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
        use_sampler: bool = False,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)
        if n_qubits_transformer > 0:
            self.transformers = nn.Sequential(
                *[TransformerBlockQuantum(embed_dim, num_heads, ffn_dim, n_qubits_transformer, n_qubits_ffn, dropout) for _ in range(num_blocks)]
            )
        else:
            self.transformers = nn.Sequential(
                *[TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout) for _ in range(num_blocks)]
            )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)
        self.use_sampler = use_sampler
        if use_sampler:
            self.sampler = QuantumSamplerQNN()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_embedding(tokens)
        x = self.transformers(x)
        x = self.dropout(x.mean(dim=1))
        logits = self.classifier(x)
        if self.use_sampler:
            probs = self.sampler(logits)
            return probs
        return logits

# ------------------------------------------------------------
# Hybrid Transformer (QML façade)
# ------------------------------------------------------------
class HybridTransformerQML(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = TextClassifierQuantum(**kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

def create_hybrid_qml_model(**kwargs) -> HybridTransformerQML:
    return HybridTransformerQML(**kwargs)
