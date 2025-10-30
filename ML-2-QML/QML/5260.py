import numpy as np
import qiskit
from qiskit import QuantumCircuit as QC, assemble, transpile
import networkx as nx
import itertools
from typing import Iterable, Sequence, List, Tuple

# ----------------------------------------------------------------------
# Quantum circuit wrapper – inspired by ClassicalQuantumBinaryClassification.py
# ----------------------------------------------------------------------
class QuantumCircuit:
    """
    Parameterised two‑qubit circuit executed on a local Aer simulator.
    The `run` method accepts a 1‑D array of angles and returns
    expectation values of Pauli‑Z for each angle.
    """
    def __init__(self, n_qubits: int, backend, shots: int):
        self._circuit = QC(n_qubits)
        all_qubits = list(range(n_qubits))
        self.theta = qiskit.circuit.Parameter("theta")
        self._circuit.h(all_qubits)
        self._circuit.barrier()
        self._circuit.ry(self.theta, all_qubits)
        self._circuit.measure_all()
        self.backend = backend
        self.shots = shots

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self._circuit, self.backend)
        qobj = assemble(compiled, shots=self.shots,
                        parameter_binds=[{self.theta: theta} for theta in thetas])
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        def expectation(count_dict):
            counts = np.array(list(count_dict.values()))
            states = np.array(list(count_dict.keys())).astype(float)
            probs = counts / self.shots
            return np.sum(states * probs)
        if isinstance(result, list):
            return np.array([expectation(item) for item in result])
        return np.array([expectation(result)])

# ----------------------------------------------------------------------
# Quantum utilities – mirror classical GraphQNN functions
# ----------------------------------------------------------------------
def state_fidelity(a: qiskit.quantum_info.Statevector,
                   b: qiskit.quantum_info.Statevector) -> float:
    return float(a.fidelity(b))

def fidelity_adjacency(states: Sequence[qiskit.quantum_info.Statevector],
                       threshold: float,
                       *, secondary: float | None = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

def random_training_data(unitary: qiskit.quantum_info.Unitary,
                        samples: int) -> List[Tuple[qiskit.quantum_info.Statevector,
                                                   qiskit.quantum_info.Statevector]]:
    dataset = []
    dim = unitary.dim
    for _ in range(samples):
        amp = np.random.normal(size=(dim,)) + 1j*np.random.normal(size=(dim,))
        amp /= np.linalg.norm(amp)
        state = qiskit.quantum_info.Statevector(amp)
        out_state = unitary @ state
        dataset.append((state, out_state))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    # Target unitary on the last layer qubits
    target_unitary = qiskit.quantum_info.random_unitary(2 ** qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)

    unitaries: List[List[qiskit.quantum_info.Unitary]] = [[]]
    for layer in range(1, len(qnn_arch)):
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        layer_ops = []
        for _ in range(num_outputs):
            op = qiskit.quantum_info.random_unitary(2 ** (num_inputs + 1))
            # For simplicity we skip register swapping and embedding
            layer_ops.append(op)
        unitaries.append(layer_ops)

    return list(qnn_arch), unitaries, training_data, target_unitary

def feedforward(qnn_arch: Sequence[int],
                unitaries: Sequence[Sequence[qiskit.quantum_info.Unitary]],
                samples: Iterable[Tuple[qiskit.quantum_info.Statevector,
                                         qiskit.quantum_info.Statevector]]) -> List[List[qiskit.quantum_info.Statevector]]:
    stored = []
    for state, _ in samples:
        layerwise = [state]
        current = state
        for layer in range(1, len(qnn_arch)):
            combined = current
            for op in unitaries[layer]:
                combined = op @ combined
            current = combined
            layerwise.append(current)
        stored.append(layerwise)
    return stored

# ----------------------------------------------------------------------
# Wrapper class for quantum graph neural network
# ----------------------------------------------------------------------
class GraphQuantumNN:
    """
    Holds a quantum graph neural network defined by a list of layer widths.
    Provides feedforward, fidelity graph construction and data generation
    utilities that mirror the classical GraphQNN interface.
    """
    def __init__(self, qnn_arch: Sequence[int], samples: int = 10):
        self.arch, self.unitaries, self.training_data, self.target_unitary = random_network(qnn_arch, samples)

    def feedforward(self,
                    samples: Iterable[Tuple[qiskit.quantum_info.Statevector,
                                             qiskit.quantum_info.Statevector]]) -> List[List[qiskit.quantum_info.Statevector]]:
        return feedforward(self.arch, self.unitaries, samples)

    def fidelity_graph(self, threshold: float,
                       *, secondary: float | None = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
        states = [state for state, _ in self.training_data]
        return fidelity_adjacency(states, threshold,
                                  secondary=secondary,
                                  secondary_weight=secondary_weight)

__all__ = [
    "QuantumCircuit",
    "state_fidelity",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "feedforward",
    "GraphQuantumNN",
]
