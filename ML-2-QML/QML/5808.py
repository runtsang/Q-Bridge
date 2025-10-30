"""
GraphQNNGen201 – Quantum implementation.

This module builds a parameterized quantum circuit per graph layer
using Qiskit.  It mirrors the classical interface, providing methods
for random unitary generation, synthetic training data, state‑wise
feed‑forward, and fidelity‑based graph construction.  The class
also exposes a lightweight fully‑connected quantum layer (FCL) for
quick experimentation.
"""

from __future__ import annotations

import itertools
import numpy as np
import qiskit
from qiskit import QuantumCircuit, transpile, Aer, execute
from qiskit.circuit import Parameter
import networkx as nx
from typing import Iterable, List, Sequence, Tuple

def _random_qubit_unitary(num_qubits: int) -> np.ndarray:
    """Generate a random Haar‑distributed unitary."""
    from scipy.stats import unitary_group
    return unitary_group.rvs(2 ** num_qubits)

def random_training_data(unitary: np.ndarray, samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate state pairs (|ψ⟩, U|ψ⟩)."""
    dataset: List[Tuple[np.ndarray, np.ndarray]] = []
    dim = unitary.shape[0]
    for _ in range(samples):
        psi = np.random.randn(dim) + 1j * np.random.randn(dim)
        psi /= np.linalg.norm(psi)
        dataset.append((psi, unitary @ psi))
    return dataset

def random_network(qnn_arch: List[int], samples: int):
    """Return architecture, list of quantum circuits per layer, training data and target unitary."""
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)

    circuits: List[List[QuantumCircuit]] = [[]]
    for layer in range(1, len(qnn_arch)):
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        layer_circuits: List[QuantumCircuit] = []
        for _ in range(num_outputs):
            qc = QuantumCircuit(num_inputs + 1)
            theta = Parameter("θ")
            qc.h(range(num_inputs + 1))
            qc.ry(theta, range(num_inputs + 1))
            qc.measure_all()
            layer_circuits.append(qc)
        circuits.append(layer_circuits)

    return qnn_arch, circuits, training_data, target_unitary

def _partial_trace(state: np.ndarray, keep: Sequence[int]) -> np.ndarray:
    """Partial trace over qubits not in ``keep`` (placeholder)."""
    # For demonstration, we simply return the state; a full implementation
    # would use tensor reshaping and partial trace operations.
    return state

def _layer_channel(
    arch: Sequence[int],
    circuits: Sequence[Sequence[QuantumCircuit]],
    layer: int,
    input_state: np.ndarray,
    backend: str = "qasm_simulator",
    shots: int = 1024,
) -> np.ndarray:
    """Apply a quantum layer and trace out auxiliary qubits."""
    num_inputs = arch[layer - 1]
    num_outputs = arch[layer]
    # Append ancillary qubits in |0⟩
    ancilla = np.zeros(2 ** (num_outputs - 1), dtype=complex)
    state = np.kron(input_state, ancilla)

    # Choose a circuit (here we pick the first for simplicity)
    qc = circuits[layer][0]
    theta_val = np.random.rand()  # Random parameter for demo
    bound_qc = qc.bind_parameters({qc.parameters[0]: theta_val})
    transpiled = transpile(bound_qc, backend=Aer.get_backend(backend))
    job = execute(transpiled, backend=Aer.get_backend(backend), shots=shots)
    result = job.result().get_counts(bound_qc)
    # Convert measurement counts to a state vector (placeholder)
    # In a real implementation, we would reconstruct the output state.
    # Here we simply return the input state for demonstration.
    return _partial_trace(state, list(range(num_inputs)))

def feedforward(
    arch: Sequence[int],
    circuits: Sequence[Sequence[QuantumCircuit]],
    samples: Iterable[Tuple[np.ndarray, np.ndarray]],
) -> List[List[np.ndarray]]:
    """Propagate each sample through the quantum graph."""
    stored_states: List[List[np.ndarray]] = []
    for sample, _ in samples:
        layerwise = [sample]
        current = sample
        for layer in range(1, len(arch)):
            current = _layer_channel(arch, circuits, layer, current)
            layerwise.append(current)
        stored_states.append(layerwise)
    return stored_states

def state_fidelity(a: np.ndarray, b: np.ndarray) -> float:
    """Squared overlap between pure states."""
    return abs(np.vdot(a, b)) ** 2

def fidelity_adjacency(
    states: Sequence[np.ndarray],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Build a weighted graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

class FCL:
    """Fully‑connected quantum layer inspired by the FCL example."""
    def __init__(self, n_qubits: int, shots: int = 1024):
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = Aer.get_backend("qasm_simulator")
        self.circuit = QuantumCircuit(n_qubits)
        self.theta = Parameter("θ")
        self.circuit.h(range(n_qubits))
        self.circuit.ry(self.theta, range(n_qubits))
        self.circuit.measure_all()

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        results = []
        for theta in thetas:
            bound = self.circuit.bind_parameters({self.theta: theta})
            transpiled = transpile(bound, backend=self.backend)
            job = execute(transpiled, backend=self.backend, shots=self.shots)
            counts = job.result().get_counts(bound)
            probs = np.array(list(counts.values())) / self.shots
            expectation = np.sum(np.array(list(counts.keys()), dtype=float) * probs)
            results.append(expectation)
        return np.array(results)

class GraphQNNGen201:
    """
    Quantum implementation of the graph‑based neural network.

    Parameters
    ----------
    arch : Sequence[int]
        Layer sizes, e.g. ``[4, 8, 4]``.
    """

    def __init__(self, arch: Sequence[int]) -> None:
        self.arch = list(arch)
        self.circuits, self.training_data, self.target = random_network(arch, samples=10)
        self.backend = "qasm_simulator"
        self.shots = 1024

    def run_layer(self, layer_idx: int, state: np.ndarray, theta: float) -> np.ndarray:
        """Execute a single quantum layer with a given parameter."""
        bound = self.circuits[layer_idx][0].bind_parameters({self.circuits[layer_idx][0].parameters[0]: theta})
        transpiled = transpile(bound, backend=Aer.get_backend(self.backend))
        job = execute(transpiled, backend=Aer.get_backend(self.backend), shots=self.shots)
        counts = job.result().get_counts(bound)
        probs = np.array(list(counts.values())) / self.shots
        expectation = np.sum(np.array(list(counts.keys()), dtype=float) * probs)
        return np.array([expectation])

    @staticmethod
    def random_network(arch: List[int], samples: int):
        """Convenience wrapper mirroring the original seed function."""
        return random_network(arch, samples)

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[np.ndarray],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ):
        """Static wrapper mirroring the original seed function."""
        return fidelity_adjacency(states, threshold, secondary=secondary, secondary_weight=secondary_weight)

    @staticmethod
    def state_fidelity(a: np.ndarray, b: np.ndarray) -> float:
        """Static wrapper for state_fidelity."""
        return state_fidelity(a, b)

    def feedforward(self, samples: Iterable[Tuple[np.ndarray, np.ndarray]]) -> List[List[np.ndarray]]:
        """Propagate samples through all quantum layers."""
        return feedforward(self.arch, self.circuits, samples)

__all__ = [
    "GraphQNNGen201",
    "random_network",
    "random_training_data",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
    "FCL",
]
