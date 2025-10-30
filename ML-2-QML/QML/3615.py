"""FraudGraphHybrid – quantum backend.

This module implements a Qiskit‑based variational circuit that encodes
classical fraud‑signal embeddings into a quantum state.  It provides
functions for generating synthetic data, evaluating the circuit,
computing fidelities, and building a fidelity‑based graph.

The design mirrors the classical module but replaces the photonic
layer operations with quantum gates.  The graph‑based similarity
information can be used as a regulariser in hybrid training pipelines.
"""

from __future__ import annotations

from typing import Iterable, Sequence, List, Tuple
import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import Statevector
import networkx as nx

# --------------------------------------------------------------------------- #
# 1. Quantum photonic‑style layer
# --------------------------------------------------------------------------- #

class FraudQuantumLayer:
    """
    A layer that maps a 2‑D classical vector into a 2‑qubit quantum state
    using a RealAmplitudes variational circuit.
    """
    def __init__(self, param_count: int = 4):
        self.param_count = param_count
        self.qc_template = RealAmplitudes(2, reps=2, entanglement="circular")

    def encode(self, vec: np.ndarray) -> Statevector:
        """Build a circuit with parameters derived from `vec` and return the statevector."""
        # Map the 2‑D vector into circuit parameters
        params = np.linspace(0, 2 * np.pi, self.param_count) * vec[0]
        qc = self.qc_template.assign_parameters(params)
        return Statevector.from_instruction(qc)

# --------------------------------------------------------------------------- #
# 2. Fidelity and adjacency utilities
# --------------------------------------------------------------------------- #

def fidelity_matrix(statevectors: Sequence[Statevector]) -> np.ndarray:
    """Return a matrix of pairwise fidelities."""
    n = len(statevectors)
    mat = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i, n):
            fid = abs(statevectors[i].data.conj().dot(statevectors[j].data)) ** 2
            mat[i, j] = fid
            mat[j, i] = fid
    return mat

def fidelity_adjacency(
    fidelity_matrix: np.ndarray,
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Same semantics as in the classical module."""
    G = nx.Graph()
    G.add_nodes_from(range(fidelity_matrix.shape[0]))
    for i in range(fidelity_matrix.shape[0]):
        for j in range(i + 1, fidelity_matrix.shape[0]):
            fid = fidelity_matrix[i, j]
            if fid >= threshold:
                G.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                G.add_edge(i, j, weight=secondary_weight)
    return G

# --------------------------------------------------------------------------- #
# 3. Synthetic data generation
# --------------------------------------------------------------------------- #

def random_training_data(samples: int) -> List[Tuple[Statevector, Statevector]]:
    """
    Generate pairs of quantum states:
    - input_state: random 2‑qubit state
    - target_state: a random unitary applied to input_state
    """
    data = []
    for _ in range(samples):
        # random input state
        dim = 4
        vec = np.random.randn(dim) + 1j * np.random.randn(dim)
        vec /= np.linalg.norm(vec)
        input_state = Statevector(vec)
        # random unitary acting on 2 qubits
        U = qiskit.quantum_info.RandomUnitary(2).data
        target_state = Statevector(U @ input_state.data)
        data.append((input_state, target_state))
    return data
