# importable quantum Python module that defines GraphQNNEstimator

from __future__ import annotations

import itertools
from typing import Iterable, Sequence, Callable, List, Tuple, Optional

import numpy as np
import networkx as nx
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, random_unitary, random_statevector, Operator

Tensor = Statevector
ScalarObservable = Callable[[Tensor], Tensor | complex | float]

def _random_qubit_state(num_qubits: int) -> Statevector:
    """Generate a random pure state for ``num_qubits``."""
    dim = 2 ** num_qubits
    vec = np.random.randn(dim) + 1j * np.random.randn(dim)
    vec /= np.linalg.norm(vec)
    return Statevector(vec)

def _random_qubit_unitary(num_qubits: int) -> QuantumCircuit:
    """Create a random unitary circuit on ``num_qubits`` qubits."""
    dim = 2 ** num_qubits
    matrix = random_unitary(dim).data
    qc = QuantumCircuit(num_qubits)
    qc.unitary(matrix, list(range(num_qubits)), label="RND")
    return qc

def random_training_data(target: QuantumCircuit, samples: int) -> List[Tuple[Statevector, Statevector]]:
    """Generate input‑output pairs for the target circuit."""
    dataset: List[Tuple[Statevector, Statevector]] = []
    for _ in range(samples):
        state = _random_qubit_state(target.num_qubits)
        out_state = state.evolve(target)
        dataset.append((state, out_state))
    return dataset

def _random_network(arch: Sequence[int], samples: int):
    """Build a random quantum network and synthetic training data."""
    num_layers = len(arch)
    circuits: List[List[QuantumCircuit]] = []

    target = QuantumCircuit(arch[-1])
    target.unitary(random_unitary(2 ** arch[-1]).data, list(range(arch[-1])), label="TGT")

    training_data = random_training_data(target, samples)

    for layer in range(1, num_layers):
        num_inputs = arch[layer - 1]
        num_outputs = arch[layer]
        layer_circuits: List[QuantumCircuit] = []
        for output in range(num_outputs):
            qc = QuantumCircuit(num_inputs + 1)
            qc.unitary(random_unitary(2 ** (num_inputs + 1)).data, list(range(num_inputs + 1)), label=f"U{layer}_{output}")
            layer_circuits.append(qc)
        circuits.append(layer_circuits)

    return list(arch), circuits, training_data, target

def _apply_and_trace(state: Statevector, qc: QuantumCircuit, keep: Sequence[int]) -> Statevector:
    """Apply a unitary and trace out qubits not in ``keep``."""
    new_state = state.evolve(qc)
    return new_state.ptrace(keep)

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
        fid = state_i.fidelity(state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

class GraphQNNEstimator:
    """Quantum graph‑based neural network estimator."""
    def __init__(self, arch: Sequence[int], circuits: List[List[QuantumCircuit]], target: QuantumCircuit):
        self.arch = list(arch)
        self.circuits = circuits
        self.target = target

    @classmethod
    def random_network(cls, arch: Sequence[int], samples: int):
        """Build a random quantum network."""
        return _random_network(arch, samples)

    def feedforward(self, inputs: Iterable[Statevector]) -> List[List[Statevector]]:
        """Propagate each input through the network."""
        activations: List[List[Statevector]] = []
        for inp in inputs:
            layerwise = [inp]
            current = inp
            for layer_idx, layer_circuits in enumerate(self.circuits):
                qc = layer_circuits[0]
                kept = list(range(1, qc.num_qubits))  # drop input qubits
                current = _apply_and_trace(current, qc, kept)
                layerwise.append(current)
            activations.append(layerwise)
        return activations

    def fidelity_adjacency(
        self,
        states: Sequence[Statevector],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Delegate to the module‑level helper."""
        return fidelity_adjacency(states, threshold, secondary=secondary, secondary_weight=secondary_weight)

    def evaluate(
        self,
        observables: Iterable[Operator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Compute expectation values for each parameter set."""
        results: List[List[complex]] = []
        for _ in parameter_sets:
            state = Statevector.from_instruction(self.target)
            results.append([state.expectation_value(obs) for obs in observables])
        return results

    def add_shots(
        self,
        results: List[List[complex]],
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """Inject Gaussian shot noise into deterministic quantum results."""
        if shots is None:
            return results
        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in results:
            noisy_row = [complex(rng.normal(mean.real, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

__all__ = [
    "GraphQNNEstimator",
    "fidelity_adjacency",
    "_random_network",
    "random_training_data",
]
