"""Quantum decoder utilities for AutoencoderGen075.

This module implements a variational quantum circuit that takes a latent
vector as input and returns a reconstructed state.  The circuit is built
with a RealAmplitudes ansatz, a swap test, and optional domain‑wall
pre‑processing.  It also provides helper functions for building a
fidelity‑based adjacency graph of latent states, mirroring the
GraphQNN utilities.
"""

from __future__ import annotations

import itertools
from typing import Iterable, Sequence, List, Tuple

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit.quantum_info import Statevector

# --------------------------------------------------------------------------- #
def _domain_wall(circuit: QuantumCircuit, a: int, b: int) -> QuantumCircuit:
    """Apply a domain wall (X gates) to qubits in the range [a, b)."""
    for i in range(a, b):
        circuit.x(i)
    return circuit

# --------------------------------------------------------------------------- #
class QuantumDecoder:
    """Variational quantum decoder circuit.

    Parameters
    ----------
    latent_dim : int
        Number of latent qubits.
    trash_dim : int, default 2
        Number of trash qubits used in the swap test.
    reps : int, default 5
        Number of repetitions of the RealAmplitudes ansatz.
    """

    def __init__(self,
                 latent_dim: int,
                 trash_dim: int = 2,
                 reps: int = 5) -> None:
        self.latent_dim = latent_dim
        self.trash_dim = trash_dim
        self.reps = reps
        self.sampler = Sampler()
        self.circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        qr = QuantumRegister(self.latent_dim + 2 * self.trash_dim + 1, 'q')
        cr = ClassicalRegister(1, 'c')
        qc = QuantumCircuit(qr, cr)

        # Domain wall on the entire register (optional)
        qc = _domain_wall(qc, 0, self.latent_dim + 2 * self.trash_dim + 1)

        # Ansatz on latent + first trash qubits
        qc.append(RealAmplitudes(self.latent_dim + self.trash_dim, reps=self.reps),
                  range(0, self.latent_dim + self.trash_dim))

        qc.barrier()

        # Swap test between latent and trash
        aux = self.latent_dim + 2 * self.trash_dim
        qc.h(aux)
        for i in range(self.trash_dim):
            qc.cswap(aux, self.latent_dim + i, self.latent_dim + self.trash_dim + i)
        qc.h(aux)

        # Measurement of auxiliary qubit
        qc.measure(aux, cr[0])
        return qc

    def reconstruct(self,
                    latent: np.ndarray | List[float],
                    param_binds: Sequence[dict] | None = None) -> np.ndarray:
        """Run the circuit for a single latent vector and return the
        expectation of the auxiliary qubit as a reconstruction vector.
        """
        if isinstance(latent, list):
            latent = np.asarray(latent, dtype=np.float64)
        # Encode latent as computational basis state: here we simply
        # ignore the encoding details and rely on the sampler.
        if param_binds is None:
            param_binds = [{p: 0.0 for p in self.circuit.parameters}]
        result = self.sampler.run(self.circuit,
                                  shots=1024,
                                  parameter_binds=param_binds)
        counts = result.get_counts()
        exp = 0.0
        for bitstring, n in counts.items():
            exp += (1 if bitstring == '1' else -1) * n
        exp /= 1024
        return np.full(self.latent_dim, exp)

    @staticmethod
    def state_fidelity(a: Statevector, b: Statevector) -> float:
        """Return the squared overlap between two pure states."""
        return abs((a.dag() @ b)[0, 0]) ** 2

    @staticmethod
    def fidelity_adjacency(states: Sequence[Statevector],
                           threshold: float,
                           *,
                           secondary: float | None = None,
                           secondary_weight: float = 0.5) -> nx.Graph:
        """Build a weighted graph from state fidelities."""
        G = nx.Graph()
        G.add_nodes_from(range(len(states)))
        for (i, ai), (j, aj) in itertools.combinations(enumerate(states), 2):
            fid = QuantumDecoder.state_fidelity(ai, aj)
            if fid >= threshold:
                G.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                G.add_edge(i, j, weight=secondary_weight)
        return G

__all__ = ['QuantumDecoder']
