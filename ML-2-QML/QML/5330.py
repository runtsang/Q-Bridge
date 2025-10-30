"""Hybrid quantum‑classical model that fuses a CNN backbone, classical transformer blocks,
and a Qiskit variational classifier head.

The architecture demonstrates how classical pre‑processing and attention mechanisms
can be combined with a quantum circuit for the final classification step.
"""

from __future__ import annotations

import math
import numpy as np

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf

from qiskit import QuantumCircuit, assemble, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit import Aer

# ---- Quantum circuit wrapper ----
class QuantumCircuitWrapper:
    """Executes a parameterised Qiskit circuit and returns expectation of Pauli‑Z."""
    def __init__(self, circuit: QuantumCircuit, backend, shots: int = 1024):
        self.circuit = circuit
        self.backend = backend
        self.shots = shots

    def run(self, params: np.ndarray) -> np.ndarray:
        compiled = transpile(self.circuit, self.backend)
        param_binds = [{p: v for p, v in zip(self.circuit.parameters, params)}]
        qobj = assemble(compiled, shots=self.shots, parameter_binds=param_binds)
        job = self.backend.run(qobj)
        result = job.result()
        counts = result.get_counts()
        if isinstance(counts, dict):
            counts = [counts]
        expectations = []
        for count in counts:
            exp = 0.0
            total = sum(count.values())
            for bitstring, cnt in count.items():
                val = 1 if bitstring[::-1][0] == '1' else -1
                exp += val * cnt / total
            expectations.append(exp)
        return np.array(expectations)

# ---- Quantum circuit builder (from ref 3) ----
def build_classifier_circuit(num_qubits: int, depth: int):
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    index = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[index], qubit)
            index += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return circuit, list(encoding), list(weights), observables

# ---- Hybrid autograd function (from ref 4) ----
class HybridFunction(torch.autograd.Function):
    """Differentiable wrapper that forwards a classical vector through a Qiskit circuit and returns
    the expectation of the first Pauli‑Z observable.  The gradient is approximated via the
    parameter‑shift rule.
    """
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit_wrapper: QuantumCircuitWrapper, shift: float = np.pi / 2):
        ctx.circuit_wrapper = circuit_wrapper
        ctx.shift = shift
        params = inputs.detach().cpu().numpy()
        expectations = ctx.circuit_wrapper.run(params)
        out = torch.tensor(expectations, dtype=inputs.dtype, device=inputs.device)
        ctx.save_for_backward(inputs, out)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        grad_inputs = torch.zeros_like(inputs)
        for i in range(inputs.shape[0]):
            shifted_plus = inputs[i] + shift
            shifted_minus = inputs[i] - shift
            exp_plus = ctx.circuit_wrapper.run(shifted_plus.detach().cpu().numpy())
            exp_minus = ctx.circuit_wrapper.run(shifted_minus.detach().cpu().numpy())
            grad_inputs[i] = (exp_plus - exp_minus) * 0.5
        return grad_inputs * grad_output, None, None

# ---- CNN backbone (identical to ml) ----
class _CNNBackbone(nn.Module):
    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(nn.Linear(16 * 7 * 7, 64), nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ---- Positional encoding (identical to ml) ----
class PositionalEncoder(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]

# ---- Transformer block (classical) ----
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

# ---- Main hybrid quantum‑classical model ----
class HybridQuantumNAT(tq.QuantumModule):
    """Full hybrid model: CNN backbone → sequence of classical transformer blocks →
    Qiskit variational classifier head."""
    def __init__(
        self,
        in_channels: int = 1,
        num_heads: int = 4,
        ffn_dim: int = 128,
        num_blocks: int = 2,
        n_qubits_classifier: int = 8,
        num_classes: int = 4,
        backend=None,
        shots: int = 1024,
    ) -> None:
        super().__init__()
        self.backbone = _CNNBackbone(in_channels)
        self.pos_encoder = PositionalEncoder(n_qubits_classifier)
        self.transformers = nn.Sequential(
            *[TransformerBlock(n_qubits_classifier, num_heads, ffn_dim, dropout=0.1) for _ in range(num_blocks)]
        )
        self.backend = backend or Aer.get_backend("aer_simulator")
        self.shots = shots
        # Variational classifier circuit (depth 0 → only encoding)
        self.classifier_circuit, self.enc_params, self.classifier_params, self.observables = build_classifier_circuit(
            n_qubits_classifier, depth=0
        )
        self.circuit_wrapper = QuantumCircuitWrapper(self.classifier_circuit, self.backend, shots=self.shots)
        # Linear layer mapping expectation to logits
        self.classifier = nn.Linear(1, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)                 # [B, n_qubits_classifier]
        x = x.unsqueeze(1)                   # [B, 1, n_qubits_classifier]
        x = self.pos_encoder(x)
        x = self.transformers(x)
        x = x.mean(dim=1)                    # [B, n_qubits_classifier]
        # Run quantum classifier for each sample
        expectations = []
        for sample in x:
            exp = self.circuit_wrapper.run(sample.detach().cpu().numpy())
            expectations.append(exp[0])      # take first observable
        expectations = torch.tensor(expectations, dtype=x.dtype, device=x.device).unsqueeze(-1)
        logits = self.classifier(expectations)
        return logits

__all__ = ["HybridQuantumNAT"]
