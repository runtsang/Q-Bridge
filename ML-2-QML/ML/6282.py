import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple
import itertools

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.primitives import Sampler as QiskitSampler
from qiskit.quantum_info import Statevector

import networkx as nx


class SamplerQNN(nn.Module):
    """Hybrid sampler that preprocesses inputs classically and samples with a variational circuit.

    The network consists of a small linear mixer followed by a Qiskit variational circuit.
    Parameters are shared between the classical and quantum parts so that the whole model
    can be trained end‑to‑end with a gradient‑based optimiser.
    """

    def __init__(self, in_features: int = 2, hidden_dim: int = 4, n_qubits: int = 2):
        super().__init__()
        # Classical embedding
        self.pre = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_qubits * 2)  # two parameters per qubit (ry)
        )
        self.n_qubits = n_qubits
        # Quantum circuit template
        self.params = ParameterVector("theta", 2 * n_qubits)
        self.qc = self._build_circuit()
        self.sampler = QiskitSampler()
        # Register the circuit for later use
        self.qc = transpile(self.qc, backend=self.sampler.backend)
        # Keep a cache of statevectors for graph utilities
        self._state_cache: List[Statevector] = []

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        # Parameterised Ry for each qubit
        for i in range(self.n_qubits):
            qc.ry(self.params[2 * i], i)
        # Entangling layer
        for i in range(self.n_qubits - 1):
            qc.cx(i, i + 1)
        # Parameterised Ry again
        for i in range(self.n_qubits):
            qc.ry(self.params[2 * i + 1], i)
        return qc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a probability distribution over all basis states."""
        # Classical preprocessing
        theta = self.pre(x)  # shape (..., 2*n_qubits)
        theta_np = theta.detach().cpu().numpy()
        # Prepare parameter values
        params = theta_np.reshape(-1, 2 * self.n_qubits)
        # Sample probabilities
        probs = self.sampler.run(self.qc, parameter_values=params).result().get_counts()
        # Convert counts to probability vector
        probs_tensor = torch.zeros((x.shape[0], 2 ** self.n_qubits))
        for i, p_dict in enumerate(probs):
            for bitstring, count in p_dict.items():
                idx = int(bitstring[::-1], 2)  # reverse because Qiskit uses little‑endian
                probs_tensor[i, idx] = count / sum(p_dict.values())
        return probs_tensor

    def sample_statevectors(self, x: torch.Tensor) -> List[Statevector]:
        """Return the statevectors for each input in the batch."""
        theta = self.pre(x).detach().cpu().numpy()
        params = theta.reshape(-1, 2 * self.n_qubits)
        svs = []
        for p in params:
            bound_qc = self.qc.bind_parameters(dict(zip(self.params, p)))
            sv = Statevector(bound_qc)
            svs.append(sv)
            self._state_cache.append(sv)
        return svs

    def fidelity_graph(self, threshold: float = 0.9, secondary: float | None = None) -> nx.Graph:
        """Create a graph of states in the cache based on fidelity."""
        return fidelity_adjacency(self._state_cache, threshold, secondary=secondary)


# Utility functions from GraphQNN
def state_fidelity(a: Statevector, b: Statevector) -> float:
    return abs((a.dag() @ b)[0, 0]) ** 2


def fidelity_adjacency(states: List[Statevector], threshold: float, *, secondary: float | None = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(a, b)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph
