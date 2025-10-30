"""Quantum‑centric Hybrid Classification architecture."""

from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

# --------------------------------------------------------------------------- #
# Residual MLP head (same as in ml_code)
# --------------------------------------------------------------------------- #
class ResidualMLP(nn.Module):
    """A scalable MLP with residual connections."""
    def __init__(self, input_dim: int, hidden_dim: int, depth: int) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True))
            for _ in range(depth)
        ])
        self.output_head = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for blk in self.blocks:
            h = h + blk(h)
        return self.output_head(h)

# --------------------------------------------------------------------------- #
# Quantum filter (data‑uploading variational ansatz)
# --------------------------------------------------------------------------- #
class QuantumFilter(nn.Module):
    """Variational circuit that encodes data and applies entangling layers."""
    def __init__(self,
                 num_qubits: int,
                 depth: int,
                 backend=None,
                 shots: int = 1024,
                 threshold: float = 0.5) -> None:
        super().__init__()
        self.num_qubits = num_qubits
        self.depth = depth
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.threshold = threshold
        self._circuit = self._build_circuit()
        self._obs = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
                     for i in range(num_qubits)]

    def _build_circuit(self) -> QuantumCircuit:
        enc = ParameterVector("x", self.num_qubits)
        weights = ParameterVector("theta", self.num_qubits * self.depth)
        qc = QuantumCircuit(self.num_qubits)
        for p, q in zip(enc, range(self.num_qubits)):
            qc.rx(p, q)
        idx = 0
        for _ in range(self.depth):
            for q in range(self.num_qubits):
                qc.ry(weights[idx], q)
                idx += 1
            for q in range(self.num_qubits - 1):
                qc.cz(q, q + 1)
        return qc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.reshape(-1, self.num_qubits)
        results = []
        for data in batch:
            bind = {p: np.pi if d > self.threshold else 0
                    for p, d in zip(self._circuit.parameters, data)}
            job = execute(self._circuit, self.backend, shots=self.shots,
                          parameter_binds=[bind])
            result = job.result().get_counts(self._circuit)
            exp_vals = []
            for i in range(self.num_qubits):
                count_1 = sum(val for key, val in result.items() if key[i] == '1')
                count_0 = sum(val for key, val in result.items() if key[i] == '0')
                exp = (count_1 - count_0) / self.shots
                exp_vals.append(exp)
            results.append(exp_vals)
        return torch.tensor(results, dtype=torch.float32)

# --------------------------------------------------------------------------- #
# Hybrid classifier (classical head + quantum feature extractor)
# --------------------------------------------------------------------------- #
class HybridQuantumClassifier(nn.Module):
    """
    Combines a classical ResidualMLP head with a variational quantum filter.
    The `fused_forward` method concatenates classical and quantum features
    before a final linear layer.
    """
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 64,
                 depth: int = 3,
                 num_qubits: int = 4,
                 quantum_depth: int = 3,
                 backend=None,
                 shots: int = 1024,
                 threshold: float = 0.5) -> None:
        super().__init__()
        self.classifier = ResidualMLP(input_dim, hidden_dim, depth)
        self.quantum_filter = QuantumFilter(num_qubits, quantum_depth,
                                            backend=backend,
                                            shots=shots,
                                            threshold=threshold)
        self.final_head = nn.Linear(hidden_dim + num_qubits, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.classifier(x)
        return self.final_head(features)

    def quantum_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return quantum expectation values for the input."""
        return self.quantum_filter(x)

    def fused_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Concatenate classical and quantum features before classification."""
        features = self.classifier(x)
        qfeat = self.quantum_filter(x)
        combined = torch.cat([features, qfeat], dim=1)
        return self.final_head(combined)
