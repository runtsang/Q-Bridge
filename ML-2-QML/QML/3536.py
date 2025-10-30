"""
Quantum‑hybrid graph neural network skeleton based on Qiskit.

The module mirrors the classical API while delegating most heavy
operations to Qiskit primitives.  It can generate random quantum
networks, compute fidelities, build adjacency graphs, and expose a
Qiskit EstimatorQNN regressor for training.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit_machine_learning.estimators import SamplerEstimator

def _random_qubit_unitary(num_qubits: int) -> Statevector:
    """Generate a random unitary as a Statevector object."""
    dim = 2 ** num_qubits
    mat = np.random.normal(size=(dim, dim)) + 1j * np.random.normal(size=(dim, dim))
    mat = np.linalg.qr(mat)[0]
    return Statevector(mat)

def _random_qubit_state(num_qubits: int) -> Statevector:
    """Generate a random pure state."""
    dim = 2 ** num_qubits
    vec = np.random.normal(size=(dim,)) + 1j * np.random.normal(size=(dim,))
    vec = vec / np.linalg.norm(vec)
    return Statevector(vec)

def random_training_data(unitary: Statevector, samples: int) -> List[Tuple[Statevector, Statevector]]:
    dataset: List[Tuple[Statevector, Statevector]] = []
    for _ in range(samples):
        state = _random_qubit_state(unitary.num_qubits)
        dataset.append((state, unitary @ state))
    return dataset

def random_network(qnn_arch: List[int], samples: int):
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)
    # For simplicity we only store the target unitary as a single “layer”.
    circuits = [[QuantumCircuit(qnn_arch[-1])]]
    circuits[0][0].unitary(target_unitary.data, list(range(qnn_arch[-1])))
    return qnn_arch, circuits, training_data, target_unitary

def state_fidelity(a: Statevector, b: Statevector) -> float:
    """Return |<a|b>|^2."""
    return abs(np.vdot(a.data, b.data)) ** 2

def fidelity_adjacency(
    states: Sequence[Statevector],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

def create_estimator_qnn(num_qubits: int = 1) -> QiskitEstimatorQNN:
    """Create a simple Qiskit EstimatorQNN circuit."""
    params = [Parameter(f"θ_{i}") for i in range(num_qubits)]
    qc = QuantumCircuit(num_qubits)
    for i, p in enumerate(params):
        qc.rx(p, i)
    observable = SparsePauliOp.from_list([("Z" * num_qubits, 1)])
    estimator = SamplerEstimator()
    return QiskitEstimatorQNN(
        circuit=qc,
        observables=observable,
        input_params=params,
        estimator=estimator,
    )

class HybridGraphQNN:
    """Quantum‑hybrid graph neural network with a public API mirroring the classical counterpart."""
    def __init__(self, arch: Sequence[int], simulator: str = "aer_simulator") -> None:
        self.arch = list(arch)
        self.simulator = simulator
        self.circuits: List[List[QuantumCircuit]] = []
        self.training_data: List[Tuple[Statevector, Statevector]] = []
        self.target_unitary: Statevector | None = None

    def random_initialize(self, samples: int = 100) -> None:
        _, self.circuits, self.training_data, self.target_unitary = random_network(self.arch, samples)

    def feedforward(self, samples: Iterable[Tuple[Statevector, Statevector]]) -> List[List[Statevector]]:
        # Placeholder: return input states unchanged
        return [[s[0]] for s in samples]

    def adjacency_graph(
        self,
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        # Use the first (and only) layer circuits as state proxies
        states_vec = [Statevector.from_instruction(circ) for circ in self.circuits[0]]
        return fidelity_adjacency(states_vec, threshold, secondary=secondary, secondary_weight=secondary_weight)

    def estimator(self) -> QiskitEstimatorQNN:
        return create_estimator_qnn(num_qubits=self.arch[0])

    def train_estimator(self, params: dict, shots: int = 1024) -> QiskitEstimatorQNN:
        qnn = self.estimator()
        qnn.set_weights(params)
        # The estimator will be fitted to the training data
        qnn.fit(self.training_data, shots=shots)
        return qnn

    def predict(self, qnn: QiskitEstimatorQNN, inputs: Statevector) -> Statevector:
        return qnn.predict(inputs)

__all__ = [
    "HybridGraphQNN",
    "create_estimator_qnn",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "fidelity_adjacency",
]
