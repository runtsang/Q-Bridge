import numpy as np
import networkx as nx
import itertools
from typing import List, Tuple, Iterable, Sequence

from qiskit import QuantumCircuit, QuantumRegister, Aer, execute
from qiskit.quantum_info import Statevector, random_statevector, random_unitary
from qiskit.opflow import StateFn, ExpectationFactory

Tensor = np.ndarray

def _random_qubit_unitary(num_qubits: int) -> QuantumCircuit:
    """Return a random unitary circuit on ``num_qubits`` qubits."""
    dim = 2 ** num_qubits
    matrix = random_unitary(dim).data
    qc = QuantumCircuit(num_qubits)
    qc.unitary(matrix, qc.qubits)
    return qc

def _random_qubit_state(num_qubits: int) -> Statevector:
    """Return a random pure state on ``num_qubits`` qubits."""
    return random_statevector(2 ** num_qubits)

def random_training_data(unitary: QuantumCircuit, samples: int) -> List[Tuple[Statevector, Statevector]]:
    dataset: List[Tuple[Statevector, Statevector]] = []
    for _ in range(samples):
        state = _random_qubit_state(unitary.num_qubits - 1)  # exclude output placeholder
        target = state.evolve(unitary)
        dataset.append((state, target))
    return dataset

def random_network(qnn_arch: List[int], samples: int):
    """Generate a random QNN architecture with corresponding unitary list."""
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)

    unitaries: List[List[QuantumCircuit]] = [[]]
    for layer in range(1, len(qnn_arch)):
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        layer_ops: List[QuantumCircuit] = []
        for output in range(num_outputs):
            qc = _random_qubit_unitary(num_inputs + 1)
            if num_outputs > 1:
                # swap the output qubit into its designated place
                qc.swap(num_inputs, num_inputs + output)
            layer_ops.append(qc)
        unitaries.append(layer_ops)

    return qnn_arch, unitaries, training_data, target_unitary

def _partial_trace_keep(state: Statevector, keep: Sequence[int]) -> Statevector:
    """Return a reduced statevector keeping the qubits in ``keep``."""
    return state.reduce(keep)

def _partial_trace_remove(state: Statevector, remove: Sequence[int]) -> Statevector:
    keep = [i for i in range(state.num_qubits) if i not in remove]
    return _partial_trace_keep(state, keep)

def _layer_channel(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[QuantumCircuit]],
    layer: int,
    input_state: Statevector,
) -> Statevector:
    """Apply the layer ``layer`` to ``input_state`` and return the reduced state."""
    num_inputs = qnn_arch[layer - 1]
    num_outputs = qnn_arch[layer]
    state = input_state
    for gate in unitaries[layer][0:]:
        state = state.evolve(gate)
    # remove input qubits to keep only outputs
    return _partial_trace_remove(state, range(num_inputs))

def feedforward(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[QuantumCircuit]],
    samples: Iterable[Tuple[Statevector, Statevector]],
) -> List[List[Statevector]]:
    stored_states: List[List[Statevector]] = []
    for sample, _ in samples:
        layerwise = [sample]
        current_state = sample
        for layer in range(1, len(qnn_arch)):
            current_state = _layer_channel(qnn_arch, unitaries, layer, current_state)
            layerwise.append(current_state)
        stored_states.append(layerwise)
    return stored_states

def state_fidelity(a: Statevector, b: Statevector) -> float:
    """Return the absolute squared overlap between pure states ``a`` and ``b``."""
    return abs((a.dag() @ b)[0, 0]) ** 2

def fidelity_adjacency(
    states: Sequence[Statevector],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Create a weighted adjacency graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

class SharedGraphQNN:
    """
    Quantum graph neural network with variational layers.
    The network is constructed from a list of random unitaries per layer.
    """
    def __init__(self, arch: Sequence[int], backend: str = "statevector_simulator"):
        self.arch = arch
        self.backend = Aer.get_backend(backend)
        self.arch, self.unitaries, self.training_data, self.target_unitary = random_network(arch, samples=10)

    def forward(self, input_state: Statevector) -> List[Statevector]:
        """Apply the network to ``input_state`` and return all layerwise states."""
        states = [input_state]
        current = input_state
        for layer in range(1, len(self.arch)):
            current = _layer_channel(self.arch, self.unitaries, layer, current)
            states.append(current)
        return states

    def fidelity_graph(self, states: List[Statevector], threshold: float) -> nx.Graph:
        """Construct a graph from state fidelities."""
        return fidelity_adjacency(states, threshold)

    @staticmethod
    def random_network(qnn_arch: Sequence[int], samples: int):
        return random_network(qnn_arch, samples)

    @staticmethod
    def random_training_data(unitary: QuantumCircuit, samples: int):
        return random_training_data(unitary, samples)

    @staticmethod
    def feedforward(
        qnn_arch: Sequence[int],
        unitaries: Sequence[Sequence[QuantumCircuit]],
        samples: Iterable[Tuple[Statevector, Statevector]],
    ) -> List[List[Statevector]]:
        return feedforward(qnn_arch, unitaries, samples)
