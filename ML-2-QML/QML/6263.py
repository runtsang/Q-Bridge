"""Hybrid quantum graph neural network that mirrors the classical :class:`GraphQNNHybrid`.

The implementation uses Qiskit and Qutip to build a QCNN‑style variational circuit
for each node.  The graph topology is encoded by pairing qubits according to
the adjacency list.  Helper functions for random network generation, forward
propagation, and fidelity‑based adjacency construction are provided.
"""

import itertools
import networkx as nx
import qutip as qt
import numpy as np
import scipy as sc
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from typing import Iterable, Sequence, List, Tuple

# ---- QCNN‑style building blocks ---------------------------------------------

def _random_unitary(num_qubits: int) -> qt.Qobj:
    """Generate a Haar‑distributed random unitary on ``num_qubits`` qubits."""
    dim = 2 ** num_qubits
    mat = sc.random.normal(size=(dim, dim)) + 1j * sc.random.normal(size=(dim, dim))
    unitary = sc.linalg.orth(mat)
    q = qt.Qobj(unitary)
    q.dims = [[2] * num_qubits, [2] * num_qubits]
    return q

def conv_gate(params: ParameterVector) -> QuantumCircuit:
    """A 2‑qubit convolution gate used in the QCNN ansatz."""
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    qc.cx(1, 0)
    qc.rz(np.pi / 2, 0)
    return qc

def pool_gate(params: ParameterVector) -> QuantumCircuit:
    """A 2‑qubit pool gate used to reduce the number of qubits."""
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc

def conv_layer(graph: nx.Graph, param_prefix: str) -> QuantumCircuit:
    """Build a convolutional layer that applies ``conv_gate`` to each edge."""
    num_qubits = graph.number_of_nodes()
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(param_prefix, length=len(graph.edges) * 3)
    idx = 0
    for (q1, q2) in graph.edges:
        gate = conv_gate(params[idx:idx + 3])
        qc.append(gate, [q1, q2])
        idx += 3
    return qc

def pool_layer(sources: Sequence[int], sinks: Sequence[int], param_prefix: str) -> QuantumCircuit:
    """Pool two qubits into one by discarding the sink qubit."""
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(param_prefix, length=len(sources) * 3)
    idx = 0
    for src, sink in zip(sources, sinks):
        gate = pool_gate(params[idx:idx + 3])
        qc.append(gate, [src, sink])
        idx += 3
    return qc

# ---- Core hybrid network -----------------------------------------------------

class QuantumGraphQNNHybrid:
    """Quantum analogue of :class:`GraphQNNHybrid`."""
    def __init__(self, qnn_arch: Sequence[int], graph: nx.Graph) -> None:
        self.graph = graph
        self.qnn_arch = qnn_arch
        # Build a variational ansatz: one conv‑pool block per layer
        self.ansatz = QuantumCircuit(graph.number_of_nodes())
        for layer in range(1, len(qnn_arch)):
            # Convolution
            self.ansatz.compose(conv_layer(graph, f"c{layer}"), inplace=True)
            # Pooling – keep only a subset of nodes
            if layer < len(qnn_arch) - 1:
                # naive pooling: pair consecutive nodes
                sources = list(range(0, graph.number_of_nodes() // 2))
                sinks   = list(range(graph.number_of_nodes() // 2, graph.number_of_nodes()))
                self.ansatz.compose(pool_layer(sources, sinks, f"p{layer}"), inplace=True)
        self.estimator = StatevectorEstimator()
        obs = SparsePauliOp.from_list([("Z" + "I" * (graph.number_of_nodes() - 1), 1)])
        self.qnn = EstimatorQNN(
            circuit=self.ansatz,
            observables=obs,
            input_params=ParameterVector("x", length=graph.number_of_nodes()),
            weight_params=self.ansatz.parameters,
            estimator=self.estimator,
        )

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Run the quantum circuit on a batch of inputs."""
        return self.qnn.predict(inputs)

def random_graph_network(qnn_arch: Sequence[int], num_nodes: int, edge_prob: float, samples: int) -> Tuple[Sequence[int], List[qt.Qobj], List[Tuple[qt.Qobj, qt.Qobj]], qt.Qobj]:
    """Generate a random graph, random unitary layers, and training data."""
    # Build random graph
    graph = nx.erdos_renyi_graph(num_nodes, edge_prob)
    # Random unitaries per layer
    unitaries: List[qt.Qobj] = [_random_unitary(num_nodes) for _ in qnn_arch]
    target_unitary = unitaries[-1]
    # Training data: random states mapped by target unitary
    training_data: List[Tuple[qt.Qobj, qt.Qobj]] = []
    for _ in range(samples):
        state = _random_unitary(num_nodes) * qt.basis(2**num_nodes, 0)
        training_data.append((state, target_unitary * state))
    return qnn_arch, unitaries, training_data, target_unitary

def feedforward(qnn_arch: Sequence[int], unitaries: Sequence[qt.Qobj], samples: Iterable[Tuple[qt.Qobj, qt.Qobj]]) -> List[List[qt.Qobj]]:
    """Propagate states through the sequence of unitaries, mimicking the classical feedforward."""
    outputs: List[List[qt.Qobj]] = []
    for state, _ in samples:
        layerwise = [state]
        current = state
        for U in unitaries:
            current = U * current
            layerwise.append(current)
        outputs.append(layerwise)
    return outputs

def state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
    """Squared overlap between two pure states."""
    return abs((a.dag() * b)[0, 0]) ** 2

def fidelity_adjacency(states: Sequence[qt.Qobj], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
    """Construct a weighted graph from fidelities of quantum states."""
    g = nx.Graph()
    g.add_nodes_from(range(len(states)))
    for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(s_i, s_j)
        if fid >= threshold:
            g.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            g.add_edge(i, j, weight=secondary_weight)
    return g

__all__ = [
    "QuantumGraphQNNHybrid",
    "random_graph_network",
    "feedforward",
    "fidelity_adjacency",
    "state_fidelity",
]
