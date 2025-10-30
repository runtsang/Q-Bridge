"""Hybrid quantum graph neural network with classifier support.

This module mirrors the classical implementation but operates on
quantum circuits.  A simple state‑vector simulator is used to obtain
layer‑wise state vectors, which can be compared to classical
activations via fidelity.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import numpy as np
import scipy as sc
from qiskit import QuantumCircuit, Aer, execute, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector, SparsePauliOp, random_unitary

def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, List[ParameterVector], List[ParameterVector], List[SparsePauliOp]]:
    """Construct a simple layered ansatz with explicit encoding and variational parameters."""
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    index = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[index], qubit)
            index += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return circuit, list(encoding), list(weights), observables

class GraphQNNHybrid:
    """Quantum graph‑based neural network with optional classifier."""

    def __init__(self, arch: Sequence[int], depth: int = 2):
        self.arch = list(arch)
        self.depth = depth
        self.unitaries = self._random_network(arch)
        self.circuit, self.encoding, self.weights, self.observables = build_classifier_circuit(len(arch[-1]), depth)

    def _random_network(self, arch: Sequence[int]) -> List[List[QuantumCircuit]]:
        """Construct a random layered ansatz for each output qubit."""
        unitaries: List[List[QuantumCircuit]] = [[]]
        for layer in range(1, len(arch)):
            num_inputs = arch[layer - 1]
            num_outputs = arch[layer]
            layer_circuits: List[QuantumCircuit] = []
            for out in range(num_outputs):
                qc = QuantumCircuit(num_inputs + 1)
                for q in range(num_inputs + 1):
                    qc.append(random_unitary(2), [q])
                if num_outputs > 1:
                    swap = QuantumCircuit(num_inputs + 1)
                    swap.swap(num_inputs, num_inputs + out)
                    qc.append(swap, range(num_inputs + 1))
                layer_circuits.append(qc)
            unitaries.append(layer_circuits)
        return unitaries

    def random_training_data(self, samples: int) -> List[Tuple[Statevector, Statevector]]:
        """Generate training pairs (input, target) for the final layer."""
        target = random_unitary(2 ** self.arch[-1])
        target_sv = Statevector(target)
        dataset: List[Tuple[Statevector, Statevector]] = []
        for _ in range(samples):
            inp_sv = Statevector.random(2 ** self.arch[0])
            dataset.append((inp_sv, target_sv))
        return dataset

    def feedforward(self, samples: Iterable[Tuple[Statevector, Statevector]]) -> List[List[Statevector]]:
        """Propagate each input through the layered ansatz."""
        stored: List[List[Statevector]] = []
        backend = Aer.get_backend("statevector_simulator")
        for inp_sv, _ in samples:
            layer_states: List[Statevector] = [inp_sv]
            current_sv = inp_sv
            for layer in range(1, len(self.arch)):
                for gate in self.unitaries[layer]:
                    qc = gate
                    qc = transpile(qc, backend)
                    job = execute(qc, backend, initial_state=current_sv.data)
                    current_sv = Statevector(job.result().get_statevector())
                layer_states.append(current_sv)
            stored.append(layer_states)
        return stored

    @staticmethod
    def state_fidelity(a: Statevector, b: Statevector) -> float:
        return abs(np.vdot(a.data, b.data)) ** 2

    @staticmethod
    def fidelity_adjacency(states: Sequence[Statevector], threshold: float,
                           *, secondary: float | None = None,
                           secondary_weight: float = 0.5) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNNHybrid.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    def hybrid_fidelity(self, classical_outputs: Sequence[np.ndarray]) -> List[float]:
        """Compare classical activations with quantum layer outputs."""
        if not classical_outputs:
            raise ValueError("No classical outputs supplied")
        quantum_states = [layer[-1] for layer in self.feedforward([(Statevector.from_label('0'*self.arch[0]), None)])]
        return [self.state_fidelity(Statevector(state), Statevector(q)) for state, q in zip(classical_outputs, quantum_states)]

__all__ = [
    "GraphQNNHybrid",
    "build_classifier_circuit",
]
