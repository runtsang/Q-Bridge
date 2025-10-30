import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZFeatureMap, RealAmplitudes
from qiskit.quantum_info import Statevector
import networkx as nx
import itertools
from typing import List, Sequence, Tuple

class QuantumFeatureMap:
    """Encodes a classical vector into a quantum state using a Z‑feature map."""
    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.base_map = ZFeatureMap(num_qubits)

    def apply(self, x: np.ndarray) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)
        qc.append(self.base_map, range(self.num_qubits))
        for i, val in enumerate(x):
            qc.ry(val, i)
        return qc

class QuantumConvolutionLayer:
    """A lightweight convolution implemented with a RealAmplitudes ansatz."""
    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.circuit = RealAmplitudes(num_qubits, reps=2)

    def apply(self, qc: QuantumCircuit, wires: List[int]) -> QuantumCircuit:
        for w in wires:
            qc.append(self.circuit, [w])
        return qc

class QuantumPoolingLayer:
    """Simple pooling via a swap test between two qubits."""
    def __init__(self):
        pass

    def apply(self, qc: QuantumCircuit, source: int, sink: int) -> QuantumCircuit:
        qc.swap(source, sink)
        return qc

class GraphQNNHybrid:
    """Quantum graph neural network that mirrors the classical GraphQNNHybrid."""
    def __init__(self, arch: Sequence[int]):
        self.arch = list(arch)
        self.layers = []
        for i in range(len(arch) - 1):
            self.layers.append(QuantumConvolutionLayer(arch[i]))
            self.layers.append(QuantumPoolingLayer())

    def feedforward(self, node_features: List[np.ndarray], adjacency: np.ndarray) -> List[Statevector]:
        """Encode each node, apply convolution‑pool layers, and return resulting states."""
        states = []
        for x in node_features:
            qc = QuantumFeatureMap(len(x)).apply(x)
            states.append(Statevector(qc))
        # Sequentially apply layers
        for layer in self.layers:
            new_states = []
            for i, state in enumerate(states):
                qc = QuantumFeatureMap(state.num_qubits).apply(state.data)
                qc = layer.apply(qc, list(range(state.num_qubits)))
                new_states.append(Statevector(qc))
            states = new_states
        return states

    def fidelity_adjacency(self,
                           states: Sequence[Statevector],
                           threshold: float,
                           *,
                           secondary: float | None = None,
                           secondary_weight: float = 0.5) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
            fid = abs((a.data.conj().T @ b.data)[0, 0]) ** 2
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    def kernel_matrix(self,
                      a: Sequence[np.ndarray],
                      b: Sequence[np.ndarray]) -> np.ndarray:
        """Simple inner‑product quantum kernel using the feature‑map ansatz."""
        states_a = [QuantumFeatureMap(len(x)).apply(x) for x in a]
        states_b = [QuantumFeatureMap(len(x)).apply(x) for x in b]
        kernel = np.zeros((len(a), len(b)))
        for i, qa in enumerate(states_a):
            for j, qb in enumerate(states_b):
                sv_a = Statevector(qa)
                sv_b = Statevector(qb)
                kernel[i, j] = abs((sv_a.data.conj().T @ sv_b.data)[0, 0]) ** 2
        return kernel

def random_network(arch: Sequence[int], samples: int):
    """Generate a random quantum network matching the architecture."""
    unitaries = []
    for in_, out in zip(arch[:-1], arch[1:]):
        dim = 2 ** in_
        matrix = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
        matrix = np.linalg.qr(matrix)[0]
        unitaries.append(matrix)
    target_unitary = unitaries[-1]
    training_data = [(np.random.randn(target_unitary.shape[0]), target_unitary @ np.random.randn(target_unitary.shape[0])) for _ in range(samples)]
    return list(arch), unitaries, training_data, target_unitary
