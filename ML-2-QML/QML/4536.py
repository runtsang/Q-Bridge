"""Quantum‑classical hybrid GraphQNN implementation using Qiskit.

The class mirrors the classical interface but uses quantum circuits for the
weight matrices.  It supports an optional EstimatorQNN ansatz and
shot‑noise injection.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.primitives import Estimator as StatevectorEstimator

# -------------------------------------------------------------------------

def _build_estimator_qnn_circuit() -> QuantumCircuit:
    """Construct the simple 1‑qubit EstimatorQNN circuit used in the original example."""
    qc = QuantumCircuit(1)
    qc.h(0)
    param1 = Parameter("θ1")
    param2 = Parameter("θ2")
    qc.ry(param1, 0)
    qc.rx(param2, 0)
    return qc

# -------------------------------------------------------------------------

def _random_qubit_unitary(num_qubits: int) -> QuantumCircuit:
    """Return a random unitary circuit on `num_qubits` qubits."""
    qc = QuantumCircuit(num_qubits)
    for _ in range(num_qubits * 2):
        q = qc.random.randint(0, num_qubits - 1)
        gate = qc.random.choice(["h", "x", "y", "z", "ry", "rz", "cx"])
        if gate == "cx":
            q2 = qc.random.randint(0, num_qubits - 1)
            while q2 == q:
                q2 = qc.random.randint(0, num_qubits - 1)
            qc.cx(q, q2)
        elif gate in ("ry", "rz"):
            theta = qc.random.uniform(-3.1416, 3.1416)
            getattr(qc, gate)(theta, q)
        else:
            getattr(qc, gate)(q)
    return qc

def random_training_data(unitary: QuantumCircuit, samples: int) -> List[Tuple[Statevector, Statevector]]:
    """Generate training pairs (|ψ⟩, U|ψ⟩)."""
    dataset: List[Tuple[Statevector, Statevector]] = []
    for _ in range(samples):
        state = Statevector.random(unitary.num_qubits)
        dataset.append((state, unitary.compose(state)))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int) -> Tuple[List[int], List[List[QuantumCircuit]], List[Tuple[Statevector, Statevector]], QuantumCircuit]:
    """Build a random QNN architecture and training data."""
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)

    unitaries: List[List[QuantumCircuit]] = [[]]
    for layer in range(1, len(qnn_arch)):
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        layer_ops: List[QuantumCircuit] = []
        for _ in range(num_outputs):
            op = _random_qubit_unitary(num_inputs + 1)
            layer_ops.append(op)
        unitaries.append(layer_ops)

    return list(qnn_arch), unitaries, training_data, target_unitary

# -------------------------------------------------------------------------

def _layer_channel(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[QuantumCircuit]], layer: int, input_state: Statevector) -> Statevector:
    """Apply the unitary for a single layer and trace out the input qubits."""
    num_inputs = qnn_arch[layer - 1]
    num_outputs = qnn_arch[layer]
    state = input_state.tensor(Statevector.from_label("0" * num_outputs))
    layer_unitary = unitaries[layer][0]
    for gate in unitaries[layer][1:]:
        layer_unitary = layer_unitary.compose(gate)
    final = layer_unitary.compose(state)
    keep = list(range(num_inputs, num_inputs + num_outputs))
    return final.partial_trace(keep)

def feedforward(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[QuantumCircuit]], samples: Iterable[Tuple[Statevector, Statevector]]) -> List[List[Statevector]]:
    """Propagate each sample through the QNN and return the list of states per layer."""
    stored_states: List[List[Statevector]] = []
    for sample, _ in samples:
        layerwise: List[Statevector] = [sample]
        current = sample
        for layer in range(1, len(qnn_arch)):
            current = _layer_channel(qnn_arch, unitaries, layer, current)
            layerwise.append(current)
        stored_states.append(layerwise)
    return stored_states

# -------------------------------------------------------------------------

def state_fidelity(a: Statevector, b: Statevector) -> float:
    """Squared overlap of two pure states."""
    return abs((a.dag() @ b)[0, 0]) ** 2

def fidelity_adjacency(states: Sequence[Statevector], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

# -------------------------------------------------------------------------

class FastBaseEstimator:
    """Quantum estimator that evaluates expectation values of supplied observables
    for a parameterised circuit.  It can optionally add shot‑noise.
    """
    def __init__(self, circuit: QuantumCircuit, shots: int | None = None, seed: int | None = None) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)
        self._estimator = StatevectorEstimator()
        self.shots = shots
        self.rng = np.random.default_rng(seed)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(self, observables: Iterable[SparsePauliOp], parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            if self.shots is not None:
                row = [complex(self.rng.normal(val.real, max(1e-6, 1 / self.shots)),
                               self.rng.normal(val.imag, max(1e-6, 1 / self.shots))) for val in row]
            results.append(row)
        return results

# -------------------------------------------------------------------------

class GraphQNNHybrid:
    """Hybrid quantum‑classical GraphQNN that mirrors the classical interface.

    Parameters
    ----------
    qnn_arch : Sequence[int]
        Layer sizes of the network.
    use_estimator_qnn : bool, default=True
        When True the last layer is implemented with the EstimatorQNN circuit.
    shots : int | None, optional
        If provided, Gaussian shot noise is added to the expectation values.
    seed : int | None, optional
        Seed for the noise generator.
    """
    def __init__(
        self,
        qnn_arch: Sequence[int],
        use_estimator_qnn: bool = True,
        shots: int | None = None,
        seed: int | None = None,
    ) -> None:
        self.arch = list(qnn_arch)
        self.use_estimator_qnn = use_estimator_qnn
        self.circuit: QuantumCircuit | None = None
        self.estimator: FastBaseEstimator | None = None

        if use_estimator_qnn:
            # Build the EstimatorQNN circuit
            self.circuit = _build_estimator_qnn_circuit()
            self.estimator = FastBaseEstimator(self.circuit, shots=shots, seed=seed)
        else:
            # Build a random circuit for the whole network
            _, unitaries, training_data, target = random_network(qnn_arch, samples=10)
            # Flatten the circuit chain
            self.circuit = QuantumCircuit(qnn_arch[-1])
            for layer_ops in unitaries[1:]:
                for op in layer_ops:
                    self.circuit = self.circuit.compose(op)
            self.estimator = FastBaseEstimator(self.circuit, shots=shots, seed=seed)

    # -------------------------------------------------------------------------

    def random_network(self, samples: int = 10) -> Tuple[List[int], List[List[QuantumCircuit]], List[Tuple[Statevector, Statevector]], QuantumCircuit]:
        """Return a fresh random network consistent with the current architecture."""
        return random_network(self.arch, samples)

    # -------------------------------------------------------------------------

    def feedforward(self, samples: Iterable[Tuple[Statevector, Statevector]]) -> List[List[Statevector]]:
        """Propagate samples through the network."""
        if self.circuit is None:
            raise RuntimeError("No circuit defined.")
        # For the EstimatorQNN case we simply evaluate the circuit directly
        # and return the final state as a placeholder.
        return [ [Statevector.from_instruction(self.circuit)] for _ in samples ]

    # -------------------------------------------------------------------------

    def state_fidelity(self, a: Statevector, b: Statevector) -> float:
        return state_fidelity(a, b)

    # -------------------------------------------------------------------------

    def fidelity_adjacency(
        self,
        states: Sequence[Statevector],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        return fidelity_adjacency(states, threshold, secondary=secondary, secondary_weight=secondary_weight)

    # -------------------------------------------------------------------------

    def evaluate(
        self,
        observables: Iterable[SparsePauliOp],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        if self.estimator is None:
            raise RuntimeError("Estimator not initialized.")
        return self.estimator.evaluate(observables, parameter_sets)

__all__ = [
    "GraphQNNHybrid",
    "FastBaseEstimator",
    "random_network",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
]
