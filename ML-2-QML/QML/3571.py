"""Hybrid quantum graph neural network with fast estimator utilities.

This quantum implementation mirrors the classical GraphQNN API using
Qiskit primitives.  It generates random unitary layers, propagates
pure states through the network, computes fidelity‑based adjacency
graphs, and offers a FastEstimator that evaluates expectation values
of parametrised circuits with optional shot noise.
"""

from __future__ import annotations

import itertools
import numpy as np
from typing import Iterable, Sequence, List, Tuple
import networkx as nx
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector, Operator, BaseOperator
from qiskit.quantum_info.operators.base_operator import BaseOperator


def _random_qubit_unitary(num_qubits: int) -> Operator:
    dim = 2 ** num_qubits
    matrix = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    # QR decomposition gives a random unitary
    q, _ = np.linalg.qr(matrix)
    return Operator(q)


def _random_qubit_state(num_qubits: int) -> Statevector:
    dim = 2 ** num_qubits
    vec = np.random.randn(dim) + 1j * np.random.randn(dim)
    vec /= np.linalg.norm(vec)
    return Statevector(vec)


def random_training_data(unitary: Operator, samples: int) -> List[Tuple[Statevector, Statevector]]:
    data = []
    for _ in range(samples):
        state = _random_qubit_state(unitary.num_qubits)
        data.append((state, unitary @ state))
    return data


def random_network(arch: Sequence[int], samples: int) -> Tuple[Sequence[int], List[List[Operator]], List[Tuple[Statevector, Statevector]], Operator]:
    """Generate a random QNN with synthetic training data."""
    target_unitary = _random_qubit_unitary(arch[-1])
    training = random_training_data(target_unitary, samples)

    unitaries: List[List[Operator]] = [[]]
    for layer in range(1, len(arch)):
        in_q = arch[layer - 1]
        out_q = arch[layer]
        layer_ops: List[Operator] = []
        for out in range(out_q):
            op = _random_qubit_unitary(in_q + 1)
            if out_q > 1:
                # Tensor with identity for remaining outputs
                op = Operator.tensor(op, Operator.identity(out_q - 1))
                # Swap the new output qubit to its correct position
                op = op.permute(list(range(in_q)) + [in_q + out] + list(range(in_q + 1, in_q + out_q)))
            layer_ops.append(op)
        unitaries.append(layer_ops)

    return arch, unitaries, training, target_unitary


def _layer_channel(
    arch: Sequence[int], unitaries: Sequence[Sequence[Operator]], layer: int, input_state: Statevector
) -> Statevector:
    in_q = arch[layer - 1]
    out_q = arch[layer]
    # Pad state with zeros for output qubits
    zero_state = _random_qubit_state(out_q)
    zero_state = Statevector.from_label("0" * out_q)
    state = Statevector.tensor(input_state, zero_state)

    # Compose all gates in the layer
    layer_unitary = unitaries[layer][0]
    for gate in unitaries[layer][1:]:
        layer_unitary = gate @ layer_unitary

    # Apply unitary
    new_state = layer_unitary @ state
    # Trace out the input qubits (keep only output qubits)
    return new_state.trace(range(in_q))


def feedforward(
    arch: Sequence[int], unitaries: Sequence[Sequence[Operator]], samples: Iterable[Tuple[Statevector, Statevector]]
) -> List[List[Statevector]]:
    """Propagate each input state through the QNN layer‑by‑layer."""
    outputs: List[List[Statevector]] = []
    for sample, _ in samples:
        layerwise = [sample]
        current = sample
        for layer in range(1, len(arch)):
            current = _layer_channel(arch, unitaries, layer, current)
            layerwise.append(current)
        outputs.append(layerwise)
    return outputs


def state_fidelity(a: Statevector, b: Statevector) -> float:
    """Squared magnitude of the overlap between two pure states."""
    return abs((a.dag() @ b).data[0]) ** 2


def fidelity_adjacency(
    states: Sequence[Statevector], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5
) -> nx.Graph:
    """Build a weighted graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, si), (j, sj) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(si, sj)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


class FastEstimator:
    """Evaluate expectation values of observables for a parametrised circuit."""

    def __init__(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, values: Sequence[float]) -> QuantumCircuit:
        if len(values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self, observables: Iterable[BaseOperator], parameter_sets: Sequence[Sequence[float]]
    ) -> List[List[complex]]:
        """Compute expectation values for each parameter set and observable."""
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results


__all__ = [
    "random_network",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
    "FastEstimator",
]
