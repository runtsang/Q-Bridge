"""Quantum transformer and fast estimator utilities built on Qiskit.

The module provides a simple quantum‑parameterised circuit that mimics the
behaviour of a transformer block through rotations and CNOT entanglement.
A FastEstimator wrapper evaluates expectation values of arbitrary Pauli
observables and optionally adds shot noise, mirroring the classical API.
"""

from __future__ import annotations

import math
from typing import Optional, Iterable, List
import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator

# --------------------------------------------------------------------------- #
#  Quantum transformer circuit
# --------------------------------------------------------------------------- #

class QuantumTransformerCircuit:
    """Parameterised circuit that encodes a token sequence and entangles qubits.

    The circuit uses RX rotations to encode each token and a ring of CNOTs to
    create a transformer‑like attention pattern.  It is intentionally
    lightweight yet sufficient to demonstrate a quantum‑centric contribution.
    """
    def __init__(self,
                 num_tokens: int,
                 n_qubits: int,
                 num_heads: int = 1,
                 embed_dim: int = 0) -> None:
        self.num_tokens = num_tokens
        self.n_qubits   = n_qubits
        self.num_heads  = num_heads
        self.embed_dim  = embed_dim

    def build(self, token_indices: List[int]) -> QuantumCircuit:
        """Return a quantum circuit that encodes the given tokens."""
        qc = QuantumCircuit(self.n_qubits)
        # Encode tokens as rotation angles
        for i, token in enumerate(token_indices):
            angle = token / 10.0  # simple linear mapping
            wire = i % self.n_qubits
            qc.rx(angle, wire)
        # Entangle qubits to simulate multi‑head attention
        for i in range(self.n_qubits - 1):
            qc.cx(i, i + 1)
        qc.cx(self.n_qubits - 1, 0)
        return qc

# --------------------------------------------------------------------------- #
#  FastEstimator quantum wrapper
# --------------------------------------------------------------------------- #

class FastEstimatorQuantum:
    """Evaluate expectation values of Pauli operators on a quantum circuit.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the circuit.
    shots : int | None, optional
        If provided, Gaussian noise with variance ``1/shots`` is added to each
        observable to emulate shot noise.
    seed : int | None, optional
        Random seed for reproducible noise.
    """
    def __init__(self,
                 n_qubits: int,
                 shots: Optional[int] = None,
                 seed: Optional[int] = None) -> None:
        self.n_qubits = n_qubits
        self.shots = shots
        self.seed = seed
        self._rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

    def evaluate(self,
                 observables: Iterable[BaseOperator],
                 parameter_sets: List[List[int]]) -> List[List[complex]]:
        """Return expectation values for each observable and parameter set."""
        results: List[List[complex]] = []
        for token_indices in parameter_sets:
            circuit = QuantumTransformerCircuit(len(token_indices), self.n_qubits).build(token_indices)
            state = Statevector.from_instruction(circuit)
            row = [state.expectation_value(obs) for obs in observables]
            # Add shot‑noise if requested
            if self.shots is not None:
                row = [complex(self._rng.normal(float(val.real), max(1e-6, 1 / self.shots)), 0) for val in row]
            results.append(row)
        return results

# --------------------------------------------------------------------------- #
#  Unified quantum estimator class
# --------------------------------------------------------------------------- #

class HybridTransformerEstimator(FastEstimatorQuantum):
    """Convenience class that bundles a quantum transformer and the FastEstimator API."""
    def __init__(self,
                 n_qubits: int,
                 shots: Optional[int] = None,
                 seed: Optional[int] = None) -> None:
        super().__init__(n_qubits, shots, seed)

__all__ = [
    "QuantumTransformerCircuit",
    "FastEstimatorQuantum",
    "HybridTransformerEstimator",
]
