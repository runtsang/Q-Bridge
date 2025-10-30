from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.primitives import Sampler as QiskitSampler
from qiskit.quantum_info import Statevector

import networkx as nx
import itertools
from typing import List, Tuple


class SamplerQNN:
    """Quantum‑only sampler that can be embedded in a hybrid model.

    The circuit contains two layers of Ry gates with an entangling CX layer.
    Parameters are exposed as a ParameterVector so that an external optimiser
    can update them.  The class provides a method to sample a batch of
    parameter vectors and to construct a fidelity graph.
    """

    def __init__(self, n_qubits: int = 2):
        self.n_qubits = n_qubits
        self.params = ParameterVector("theta", 2 * n_qubits)
        self.qc = self._build_circuit()
        self.sampler = QiskitSampler()
        self.qc = transpile(self.qc, backend=self.sampler.backend)
        self._state_cache: List[Statevector] = []

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            qc.ry(self.params[2 * i], i)
        for i in range(self.n_qubits - 1):
            qc.cx(i, i + 1)
        for i in range(self.n_qubits):
            qc.ry(self.params[2 * i + 1], i)
        return qc

    def sample(self, params: List[List[float]]) -> List[dict]:
        """Return measurement counts for each parameter set."""
        results = []
        for p in params:
            bound_qc = self.qc.bind_parameters(dict(zip(self.params, p)))
            res = self.sampler.run(bound_qc).result()
            results.append(res.get_counts())
        return results

    def statevector(self, params: List[float]) -> Statevector:
        bound_qc = self.qc.bind_parameters(dict(zip(self.params, params)))
        sv = Statevector(bound_qc)
        self._state_cache.append(sv)
        return sv

    def fidelity_graph(self, threshold: float = 0.9, secondary: float | None = None) -> nx.Graph:
        return fidelity_adjacency(self._state_cache, threshold, secondary=secondary)


# Utility functions
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
