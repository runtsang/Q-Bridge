"""GraphQNNGen116 – quantum implementation.

This module mirrors the classical version but replaces linear layers with
parameterised quantum circuits.  The public API is identical so that the
class can be swapped at runtime.  The implementation uses Qiskit
and keeps the same helper functions that were present in the original
seed code.

Key extensions:
* A quantum‑style fully‑connected layer that executes a single‑qubit
  Ry rotation per parameter.
* A quanvolution filter that applies a random circuit to each patch.
* All fidelity and adjacency utilities are adapted to work with
  quantum states.
"""
from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import Statevector

Tensor = np.ndarray  # alias for readability


def _random_qubit_unitary(num_qubits: int) -> Statevector:
    """Generate a Haar‑random unitary on `num_qubits` qubits."""
    dim = 2 ** num_qubits
    matrix = np.random.normal(size=(dim, dim)) + 1j * np.random.normal(size=(dim, dim))
    q, _ = np.linalg.qr(matrix)
    return Statevector(q)


def _random_qubit_state(num_qubits: int) -> Statevector:
    """Sample a random pure state on `num_qubits` qubits."""
    dim = 2 ** num_qubits
    vec = np.random.normal(size=dim) + 1j * np.random.normal(size=dim)
    vec /= np.linalg.norm(vec)
    return Statevector(vec)


def random_training_data(unitary: Statevector, samples: int) -> List[Tuple[Statevector, Statevector]]:
    """Create a dataset of states and their images under `unitary`."""
    dataset: List[Tuple[Statevector, Statevector]] = []
    num_qubits = int(np.log2(unitary.dim))
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        dataset.append((state, unitary.evolve(state)))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int) -> Tuple[List[int], List[List[Statevector]], List[Tuple[Statevector, Statevector]], Statevector]:
    """Build a random quantum network and a training set for its last layer."""
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)

    unitaries: List[List[Statevector]] = [[]]
    for layer in range(1, len(qnn_arch)):
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        layer_ops: List[Statevector] = []
        for _ in range(num_outputs):
            op = _random_qubit_unitary(num_inputs + 1)
            layer_ops.append(op)
        unitaries.append(layer_ops)

    return list(qnn_arch), unitaries, training_data, target_unitary


def _partial_trace_keep(state: Statevector, keep: Sequence[int]) -> Statevector:
    """Return the partial trace over all qubits *except* those in `keep`."""
    return state.partial_trace(keep)


def _partial_trace_remove(state: Statevector, remove: Sequence[int]) -> Statevector:
    """Return the state after tracing out qubits in `remove`."""
    keep = [i for i in range(state.num_qubits) if i not in remove]
    return _partial_trace_keep(state, keep)


def _layer_channel(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[Statevector]], layer: int, input_state: Statevector) -> Statevector:
    """Apply the `layer`‑th set of unitaries to `input_state` and trace out the input qubits."""
    num_inputs = qnn_arch[layer - 1]
    num_outputs = qnn_arch[layer]
    zero_state = Statevector.from_label("0" * num_outputs)
    joint = input_state.tensor(zero_state)

    unitary = unitaries[layer][0]
    for gate in unitaries[layer][1:]:
        unitary = gate @ unitary
    joint = unitary.evolve(joint)

    return _partial_trace_remove(joint, list(range(num_inputs)))


def feedforward(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[Statevector]], samples: Iterable[Tuple[Statevector, Statevector]]) -> List[List[Statevector]]:
    """Propagate a batch of samples through the quantum network."""
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
    """Squared absolute overlap of two pure states."""
    return abs(np.vdot(a.data, b.data)) ** 2


def fidelity_adjacency(states: Sequence[Statevector], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
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


class FCLQuantum:
    """Quantum analogue of the fully‑connected layer – a single‑qubit Ry rotation per parameter."""

    def __init__(self, n_qubits: int, backend=None, shots: int = 1024):
        self.n_qubits = n_qubits
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.circuit = QuantumCircuit(n_qubits)
        self.theta = [Parameter(f"theta{i}") for i in range(n_qubits)]
        self.circuit.h(range(n_qubits))
        self.circuit.barrier()
        for i, t in enumerate(self.theta):
            self.circuit.ry(t, i)
        self.circuit.measure_all()

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        param_binds = [{t: val for t, val in zip(self.theta, thetas)}]
        job = execute(self.circuit, self.backend, shots=self.shots, parameter_binds=param_binds)
        result = job.result().get_counts(self.circuit)
        counts = np.array(list(result.values()))
        probs = counts / self.shots
        expectation = np.sum(np.array(list(result.keys()), dtype=float) * probs)
        return np.array([expectation])


class ConvQuantum:
    """Quantum quanvolution filter – a random circuit applied to each qubit of a patch."""

    def __init__(self, kernel_size: int, backend=None, shots: int = 1024, threshold: float = 0.5):
        self.n_qubits = kernel_size ** 2
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.threshold = threshold
        self.circuit = QuantumCircuit(self.n_qubits)
        self.theta = [Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self.circuit.rx(self.theta[i], i)
        self.circuit.barrier()
        self.circuit += random_circuit(self.n_qubits, 2)
        self.circuit.measure_all()

    def run(self, data: np.ndarray) -> float:
        """Return the average probability of measuring |1> across all qubits."""
        data = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)

        job = execute(self.circuit, self.backend, shots=self.shots, parameter_binds=param_binds)
        result = job.result().get_counts(self.circuit)

        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val

        return counts / (self.shots * self.n_qubits)


class GraphQNNGen116:
    """Quantum‑style graph neural network.

    The API matches the classical version but the underlying
    computation uses Qiskit.  The network is defined by a list of
    qubit counts per layer.  The final layer is trained to approximate
    a target unitary.
    """

    def __init__(self, arch: Sequence[int], mode: str = "quantum") -> None:
        if mode!= "quantum":
            raise ValueError("Quantum implementation only supports mode='quantum'.")
        self.arch = list(arch)
        self.mode = mode
        self.unitaries: List[List[Statevector]] | None = None

    def build(self, unitaries: Sequence[Sequence[Statevector]]) -> None:
        """Attach a pre‑generated set of unitaries to the instance."""
        self.unitaries = [list(layer) for layer in unitaries]

    def forward(self, state: Statevector) -> Statevector:
        """Propagate a single input state through the network."""
        if self.unitaries is None:
            raise RuntimeError("Unitary set not initialized. Call `build` or use `random_network`.")
        current = state
        for layer in range(1, len(self.arch)):
            current = _layer_channel(self.arch, self.unitaries, layer, current)
        return current

    def run(self, data: np.ndarray) -> np.ndarray:
        """Convenience wrapper that accepts a numpy array and returns a statevector."""
        state = Statevector.from_array(data)
        return self.forward(state).data

    def feedforward(self, samples: Iterable[Tuple[Statevector, Statevector]]) -> List[List[Statevector]]:
        """Run a batch of samples and return layerwise states."""
        return feedforward(self.arch, self.unitaries, samples)

    @staticmethod
    def random_network(arch: Sequence[int], samples: int) -> Tuple[List[int], List[List[Statevector]], List[Tuple[Statevector, Statevector]], Statevector]:
        """Return a fully random network together with a training set."""
        return random_network(arch, samples)

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[Statevector],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        return fidelity_adjacency(states, threshold, secondary=secondary, secondary_weight=secondary_weight)


__all__ = [
    "GraphQNNGen116",
    "FCLQuantum",
    "ConvQuantum",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "fidelity_adjacency",
]
