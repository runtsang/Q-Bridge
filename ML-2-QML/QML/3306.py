"""GraphQNNGen: quantum implementation of a graph‑neural‑network.

The module mirrors the classical counterpart but uses Qiskit
to build and evaluate variational circuits.  The public API is
compatible with the classical version: ``random_network``,
``feedforward`` and ``fidelity_adjacency``.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import numpy as np
import qiskit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import Estimator as QiskitEstimator
import qutip as qt
import scipy as sc

# ----- Quantum utilities ---------------------------------------------------

def _random_qubit_unitary(num_qubits: int) -> qt.Qobj:
    dim = 2 ** num_qubits
    matrix = sc.random.normal(size=(dim, dim)) + 1j * sc.random.normal(size=(dim, dim))
    unitary = sc.linalg.orth(matrix)
    return qt.Qobj(unitary)

def random_training_data(unitary: qt.Qobj, samples: int) -> List[Tuple[qt.Qobj, qt.Qobj]]:
    dataset: List[Tuple[qt.Qobj, qt.Qobj]] = []
    num_qubits = len(unitary.dims[0])
    for _ in range(samples):
        state = qt.rand_ket(2 ** num_qubits)
        dataset.append((state, unitary * state))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)

    total_qubits = sum(qnn_arch)
    total_params = 2 * total_qubits
    params = [Parameter(f"p{i}") for i in range(total_params)]

    qc = qiskit.QuantumCircuit(total_qubits)
    for q in range(total_qubits):
        qc.ry(params[q], q)
        qc.rz(params[total_qubits + q], q)

    observable = SparsePauliOp.from_list([("Z" * qc.num_qubits, 1)])
    estimator = QiskitEstimator()
    estimator_qnn = QiskitEstimatorQNN(
        circuit=qc,
        observables=observable,
        input_params=params[:total_qubits],
        weight_params=params[total_qubits:],
        estimator=estimator,
    )
    return qnn_arch, estimator_qnn, training_data, target_unitary

def feedforward(
    qnn_arch: Sequence[int],
    estimator_qnn: QiskitEstimatorQNN,
    samples: Iterable[Tuple[qt.Qobj, qt.Qobj]],
) -> List[List[qt.Qobj]]:
    stored: List[List[qt.Qobj]] = []
    for state, _ in samples:
        vec = np.array(state.data).flatten()
        sv = Statevector(vec)
        sv = sv.compose(estimator_qnn.circuit)
        out_state = qt.Qobj(sv.data.reshape(-1, 1))
        out_state.dims = [[2] * sum(qnn_arch), [1] * sum(qnn_arch)]
        stored.append([state, out_state])
    return stored

def state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
    return abs((a.dag() * b)[0, 0]) ** 2

def fidelity_adjacency(
    states: Sequence[qt.Qobj],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(a, b)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

# ----- Hybrid class --------------------------------------------------------

class GraphQNN:
    """Quantum‑only GraphQNN implementation.

    The class exposes the same high‑level API as the classical
    counterpart: ``random_network``, ``feedforward`` and
    ``fidelity_adjacency``.  It internally uses Qiskit’s
    ``EstimatorQNN`` to evaluate the variational circuit on
    arbitrary input states.
    """
    def __init__(self, arch: Sequence[int]) -> None:
        self.arch = list(arch)
        self.qnn_arch, self.estimator_qnn, self.training_data, self.target_unitary = random_network(self.arch, 10)

    def random_network(self, samples: int):
        return random_network(self.arch, samples)

    def feedforward(
        self,
        samples: Iterable[Tuple[qt.Qobj, qt.Qobj]],
    ) -> List[List[qt.Qobj]]:
        return feedforward(self.arch, self.estimator_qnn, samples)

    def fidelity_adjacency(
        self,
        states: Sequence[qt.Qobj],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        return fidelity_adjacency(states, threshold, secondary=secondary, secondary_weight=secondary_weight)

__all__ = [
    "GraphQNN",
    "random_network",
    "feedforward",
    "fidelity_adjacency",
]
