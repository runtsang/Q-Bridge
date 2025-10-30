"""Hybrid transformer implemented with Qiskit and a lightweight estimator.

Features:
- QuantumSelfAttention and QuantumFeedForward use a parametric circuit
  per token, providing a simple variational self‑attention mechanism.
- QuantumTransformer is a stack of these blocks with a classical
  linear classifier on top.
- FastBaseEstimator evaluates a list of quantum circuits and returns
  expectation values for arbitrary observables.
"""

import numpy as np
import qiskit as qk
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from typing import Iterable, Sequence, List, Optional

# ---- Quantum primitives ---------------------------------------------

class QuantumSelfAttention:
    """Quantum self‑attention built solely with Qiskit."""
    def __init__(self, n_qubits: int, backend: Optional[qk.AerSimulator] = None):
        self.n_qubits = n_qubits
        self.backend = backend or qk.Aer.get_backend("statevector_simulator")
        self.circuit = QuantumCircuit(n_qubits, n_qubits)
        self.params = [Parameter(f"θ_{i}") for i in range(n_qubits * 3)]
        self._build_circuit()

    def _build_circuit(self):
        for i in range(self.n_qubits):
            self.circuit.rx(self.params[3 * i], i)
            self.circuit.ry(self.params[3 * i + 1], i)
            self.circuit.rz(self.params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            self.circuit.cx(i, i + 1)
        self.circuit.cx(self.n_qubits - 1, 0)

    def forward(self, token: np.ndarray) -> np.ndarray:
        """Encode a single token vector into the circuit and return expectation of Z."""
        param_map = {self.params[i]: token[i] for i in range(min(len(token), self.n_qubits * 3))}
        bound = self.circuit.assign_parameters(param_map, inplace=False)
        state = Statevector(bound, backend=self.backend)
        return np.array([state.expectation_value(qk.PauliZ(j)) for j in range(self.n_qubits)])


class QuantumFeedForward:
    """Quantum feed‑forward realised by a variational circuit."""
    def __init__(self, n_qubits: int, backend: Optional[qk.AerSimulator] = None):
        self.n_qubits = n_qubits
        self.backend = backend or qk.Aer.get_backend("statevector_simulator")
        self.circuit = QuantumCircuit(n_qubits, n_qubits)
        self.params = [Parameter(f"φ_{i}") for i in range(n_qubits)]
        self._build_circuit()

    def _build_circuit(self):
        for i in range(self.n_qubits):
            self.circuit.ry(self.params[i], i)
            for j in range(i + 1, self.n_qubits):
                self.circuit.cx(i, j)

    def forward(self, token: np.ndarray) -> np.ndarray:
        param_map = {self.params[i]: token[i] for i in range(min(len(token), self.n_qubits))}
        bound = self.circuit.assign_parameters(param_map, inplace=False)
        state = Statevector(bound, backend=self.backend)
        return np.array([state.expectation_value(qk.PauliZ(j)) for j in range(self.n_qubits)])


# ---- Transformer block ----------------------------------------------

class QuantumTransformerBlock:
    """Single transformer block composed of quantum self‑attention and feed‑forward."""
    def __init__(self, embed_dim: int, n_qubits: int, backend: Optional[qk.AerSimulator] = None):
        self.embed_dim = embed_dim
        self.n_qubits = n_qubits
        self.backend = backend
        self.attn = QuantumSelfAttention(n_qubits, backend=self.backend)
        self.ffn = QuantumFeedForward(n_qubits, backend=self.backend)

    def forward(self, sequence: np.ndarray) -> np.ndarray:
        """sequence shape (seq_len, embed_dim)"""
        seq_len = sequence.shape[0]
        attn_out = np.stack([self.attn.forward(sequence[i]) for i in range(seq_len)], axis=0)
        ffn_out = np.stack([self.ffn.forward(attn_out[i]) for i in range(seq_len)], axis=0)
        return ffn_out


# ---- Transformer model -----------------------------------------------

class QuantumTransformer:
    """Transformer‑style classifier using only quantum blocks."""
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_blocks: int,
        num_classes: int,
        n_qubits: int = 8,
        backend: Optional[qk.AerSimulator] = None,
    ):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        self.num_classes = num_classes
        self.backend = backend or qk.Aer.get_backend("statevector_simulator")
        # Simple lookup table for token embeddings
        self.token_lookup = np.random.randn(vocab_size, embed_dim) * 0.1
        self.blocks = [
            QuantumTransformerBlock(embed_dim, n_qubits, backend=self.backend)
            for _ in range(num_blocks)
        ]
        self.classifier_weights = np.random.randn(embed_dim, num_classes)

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        seq = self.token_lookup[token_ids]
        for blk in self.blocks:
            seq = blk.forward(seq)
        logits = seq.mean(axis=0) @ self.classifier_weights
        return logits


# ---- Estimator -------------------------------------------------------

class FastBaseEstimator:
    """Evaluates expectation values of a list of quantum circuits."""
    def __init__(self, circuit: QuantumCircuit):
        self.circuit = circuit
        self.backend = qk.Aer.get_backend("statevector_simulator")

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        results: List[List[complex]] = []
        for params in parameter_sets:
            bound = self.circuit.assign_parameters(
                dict(zip(self.circuit.parameters, params)), inplace=False
            )
            state = Statevector(bound, backend=self.backend)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results


# Alias for compatibility with reference pair 2
FastQuantumEstimator = FastBaseEstimator

__all__ = [
    "QuantumSelfAttention",
    "QuantumFeedForward",
    "QuantumTransformerBlock",
    "QuantumTransformer",
    "FastBaseEstimator",
    "FastQuantumEstimator",
]
