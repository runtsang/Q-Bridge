import itertools
from typing import List, Tuple, Iterable, Sequence

import numpy as np
import networkx as nx

from qiskit import QuantumCircuit, QuantumRegister, Aer, execute
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector, RandomUnitary, Operator
from qiskit.opflow import StateFn, CircuitStateFn, Statevector as OpStatevector


def _random_unitary(num_qubits: int) -> np.ndarray:
    """Return a random unitary matrix of size 2^n."""
    return RandomUnitary(2 ** num_qubits).data


def _random_state(num_qubits: int) -> Statevector:
    """Return a random pure state."""
    return Statevector.random(2 ** num_qubits)


def random_training_data(unitary: np.ndarray, samples: int) -> List[Tuple[Statevector, Statevector]]:
    """Generate (input, target) pairs where target = unitary * input."""
    dataset: List[Tuple[Statevector, Statevector]] = []
    for _ in range(samples):
        inp = _random_state(len(unitary) // 2).data
        target = Operator(unitary) @ Statevector(inp)
        dataset.append((Statevector(inp), target))
    return dataset


def random_network(qnn_arch: List[int], samples: int):
    """Create a random variational network and training data."""
    target_unitary = _random_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)

    circuits: List[List[QuantumCircuit]] = [[]]
    for layer in range(1, len(qnn_arch)):
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        layer_circuits: List[QuantumCircuit] = []
        for output in range(num_outputs):
            qubits = QuantumRegister(num_inputs + 1, f"q{layer}_{output}")
            circ = QuantumCircuit(qubits)
            # Random single‑qubit rotations followed by a CNOT grid
            for q in range(num_inputs + 1):
                circ.rx(np.random.uniform(0, 2 * np.pi), qubits[q])
                circ.ry(np.random.uniform(0, 2 * np.pi), qubits[q])
                circ.rz(np.random.uniform(0, 2 * np.pi), qubits[q])
            for q in range(num_inputs):
                circ.cx(qubits[q], qubits[q + 1])
            layer_circuits.append(circ)
        circuits.append(layer_circuits)

    return qnn_arch, circuits, training_data, target_unitary


def _partial_trace(state: Statevector, keep: Sequence[int]) -> Statevector:
    """Return the reduced state vector over the specified qubits."""
    return state.reduced_state(keep)


def feedforward(
    qnn_arch: Sequence[int],
    circuits: Sequence[Sequence[QuantumCircuit]],
    samples: Iterable[Tuple[Statevector, Statevector]],
) -> List[List[Statevector]]:
    stored_states: List[List[Statevector]] = []
    for inp, _ in samples:
        layerwise = [inp]
        current = inp
        for layer in range(1, len(qnn_arch)):
            # Use the first circuit of the layer as the channel
            circ = circuits[layer][0]
            new_state = Statevector.from_instruction(circ) @ current
            # Keep only the target qubits for the next layer
            keep = list(range(qnn_arch[layer]))
            current = _partial_trace(new_state, keep)
            layerwise.append(current)
        stored_states.append(layerwise)
    return stored_states


def state_fidelity(a: Statevector, b: Statevector) -> float:
    """Return the absolute squared overlap between pure states a and b."""
    return abs(a.overlap(b)) ** 2


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


__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
]
