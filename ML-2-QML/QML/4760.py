"""Quantum implementation of QuantumClassifierModel.

This module builds a convolutional network followed by a parameterised
quantum circuit acting as the classification head.  It supports automatic
differentiation via the parameter‑shift rule and can be executed on any
Aer or Braket backend.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from qiskit import QuantumCircuit, transpile, assemble
from qiskit.circuit import ParameterVector
from qiskit.providers.aer import AerSimulator
from qiskit.quantum_info import SparsePauliOp
from typing import Iterable, Tuple, List

# ------------------------------------------------------------
# Utility: classical‑quantum hybrid layer
# ------------------------------------------------------------
class HybridFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, backend: AerSimulator,
                shots: int, shift: float) -> torch.Tensor:
        """
        Forward pass that evaluates the expectation value of Z on each qubit
        after the circuit has been executed with the provided parameters.
        """
        ctx.shift = shift
        ctx.circuit = circuit
        ctx.backend = backend
        ctx.shots = shots

        expectations = []
        for sample in inputs.detach().cpu().numpy():
            param_binds = [{circuit.params[i]: sample[i] for i in range(len(sample))}]
            compiled = transpile(circuit, backend)
            qobj = assemble(compiled, parameter_binds=param_binds, shots=shots)
            result = backend.run(qobj).result()
            counts = result.get_counts()
            exp = 0.0
            for bitstring, cnt in counts.items():
                val = (-1) ** (bitstring.count("1"))
                exp += val * cnt
            exp /= shots
            expectations.append(exp)
        return torch.tensor(expectations, dtype=torch.float32, device=inputs.device)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Parameter‑shift rule to compute gradients w.r.t the circuit parameters.
        """
        shift = ctx.shift
        grads = []
        for param in ctx.circuit.parameters:
            pos_bind = {param: shift}
            neg_bind = {param: -shift}
            compiled = transpile(ctx.circuit, ctx.backend)
            qobj_pos = assemble(compiled, parameter_binds=[pos_bind], shots=ctx.shots)
            qobj_neg = assemble(compiled, parameter_binds=[neg_bind], shots=ctx.shots)
            res_pos = ctx.backend.run(qobj_pos).result().get_counts()
            res_neg = ctx.backend.run(qobj_neg).result().get_counts()
            exp_pos = sum([(-1) ** (b.count("1")) * c for b, c in res_pos.items()]) / ctx.shots
            exp_neg = sum([(-1) ** (b.count("1")) * c for b, c in res_neg.items()]) / ctx.shots
            grads.append((exp_pos - exp_neg) / (2 * shift))
        grads = torch.tensor(grads, dtype=torch.float32, device=grad_output.device)
        return grads * grad_output, None, None, None, None

class Hybrid(nn.Module):
    """
    Wraps a parameterised quantum circuit and exposes it as a PyTorch module.
    """
    def __init__(self, n_qubits: int, backend: AerSimulator, shots: int = 1024,
                 shift: float = np.pi / 2) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots
        self.shift = shift
        self.circuit = self._build_ansatz(n_qubits)

    def _build_ansatz(self, n_qubits: int) -> QuantumCircuit:
        """
        Data‑re‑uploading ansatz with alternating Ry and CZ layers.
        """
        params = ParameterVector("theta", length=n_qubits * 3)
        circuit = QuantumCircuit(n_qubits)
        for q in range(n_qubits):
            circuit.ry(params[q], q)
        for q in range(n_qubits - 1):
            circuit.cz(q, q + 1)
        for q in range(n_qubits):
            circuit.ry(params[n_qubits + q], q)
        for q in range(n_qubits):
            circuit.ry(params[2 * n_qubits + q], q)
        return circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(x, self.circuit, self.backend,
                                    self.shots, self.shift)

# ------------------------------------------------------------
# SamplerQNN
# ------------------------------------------------------------
def SamplerQNN() -> nn.Module:
    """
    Returns a simple neural sampler that can be used in place of the
    quantum expectation head when a classical approximation is desired.
    """
    class _Sampler(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 4),
                nn.Tanh(),
                nn.Linear(4, 2),
                nn.Softmax(dim=-1),
            )

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.net(inputs)
    return _Sampler()

# ------------------------------------------------------------
# Main model
# ------------------------------------------------------------
class QuantumClassifierModel(nn.Module):
    """
    Convolutional feature extractor followed by a hybrid quantum head.
    Mirrors the public API of the classical baseline.
    """
    def __init__(self, n_qubits: int = 2, backend: AerSimulator | None = None,
                 shots: int = 1024, shift: float = np.pi / 2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.5),
        )
        dummy_input = torch.zeros(1, 3, 32, 32)
        dummy_out = self.features(dummy_input)
        flat_dim = dummy_out.numel() // dummy_out.shape[0]
        self.classifier = nn.Linear(flat_dim, n_qubits)
        self.hybrid = Hybrid(n_qubits, backend or AerSimulator(), shots, shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        out = self.hybrid(x)
        probs = torch.sigmoid(out)
        return torch.stack([probs, 1 - probs], dim=-1)

def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, List[int], List[int], List[SparsePauliOp]]:
    """
    Build a parametric quantum circuit that can be used as a stand‑alone classifier.
    Returns the circuit, the list of encoding indices, the list of parameter indices,
    and the measurement observables.
    """
    encoding = list(range(num_qubits))
    weights = ParameterVector("theta", length=num_qubits * depth)
    circuit = QuantumCircuit(num_qubits)
    for i, qubit in enumerate(encoding):
        circuit.ry(weights[i], qubit)
    index = num_qubits
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[index], qubit)
            index += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)
    observables = [
        SparsePauliOp("Z" + "I" * (num_qubits - 1)),
        SparsePauliOp("I" + "Z" + "I" * (num_qubits - 2)),
        SparsePauliOp("I" * 2 + "Z" * (num_qubits - 2)),
    ]
    return circuit, encoding, list(weights), observables

__all__ = ["QuantumClassifierModel", "build_classifier_circuit", "SamplerQNN"]
