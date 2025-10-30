from __future__ import annotations

import itertools
from typing import Iterable, Sequence, List, Tuple

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import RX, CX
from qiskit.quantum_info import Statevector, Operator


class GraphQNN:
    """Quantum graph neural network built with Qiskit.

    The architecture is a sequence of layer sizes; each layer is a
    parameterized unitary acting on the number of qubits equal to the
    output size of the layer. The final unitary is used as the target
    for synthetic training data.
    """

    def __init__(self, arch: Sequence[int]):
        self.arch = list(arch)
        self.parameters: List[Parameter] = []
        self.layers: List[QuantumCircuit] = []

        # Build a simple parameterised circuit for each layer
        for layer_idx, out_size in enumerate(self.arch[1:], start=1):
            layer_circ = QuantumCircuit(max(self.arch))
            for qubit in range(out_size):
                p = Parameter(f'theta_{layer_idx}_{qubit}')
                self.parameters.append(p)
                layer_circ.append(RX(p), [qubit])
                if layer_idx > 1:
                    for prev in range(self.arch[layer_idx - 1]):
                        layer_circ.cx(prev, qubit)
            self.layers.append(layer_circ)

        # Concatenate layers into a full circuit
        self.circuit = QuantumCircuit(max(self.arch))
        for layer_circ in self.layers:
            self.circuit.append(layer_circ, range(len(layer_circ.qubits)))

    @staticmethod
    def random_network(arch: Sequence[int], samples: int):
        """Generate a random target unitary and synthetic training data."""
        model = GraphQNN(arch)
        # Randomize parameters
        for p in model.parameters:
            p.set_value(np.random.rand())
        # Final unitary
        target_unitary = Operator(model.circuit)
        dataset: List[Tuple[Statevector, Statevector]] = []
        for _ in range(samples):
            state = Statevector.from_label('0' * arch[-1])
            transformed = Statevector(target_unitary @ state.data)
            dataset.append((state, transformed))
        return list(arch), model.layers, dataset, target_unitary

    @staticmethod
    def feedforward(
        arch: Sequence[int],
        layers: Sequence[QuantumCircuit],
        samples: Iterable[Tuple[Statevector, Statevector]],
    ) -> List[List[Statevector]]:
        """Return statevector after each layer for each sample."""
        outputs: List[List[Statevector]] = []
        for state, _ in samples:
            layer_states = [state]
            current = state
            for layer_circ in layers:
                current = current.evolve(layer_circ)
                layer_states.append(current)
            outputs.append(layer_states)
        return outputs

    @staticmethod
    def state_fidelity(a: Statevector, b: Statevector) -> float:
        """Squared overlap of pure states."""
        return abs((a.dag() @ b)[0, 0]) ** 2

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[Statevector],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Create weighted graph from state fidelities."""
        G = nx.Graph()
        G.add_nodes_from(range(len(states)))
        for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNN.state_fidelity(s_i, s_j)
            if fid >= threshold:
                G.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                G.add_edge(i, j, weight=secondary_weight)
        return G

    def evaluate(
        self,
        observables: Iterable[Operator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Compute expectation values for each parameter set and observable."""
        if not observables:
            observables = [Operator(np.eye(2 ** self.arch[-1]))]
        results: List[List[complex]] = []
        for params in parameter_sets:
            bound_circuit = self.circuit.bind_parameters(dict(zip(self.parameters, params)))
            state = Statevector.from_instruction(bound_circuit)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

    def evaluate_with_shots(
        self,
        observables: Iterable[Operator],
        parameter_sets: Sequence[Sequence[float]],
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """Add shot noise to expectation values."""
        raw = self.evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in raw:
            noisy_row = [
                rng.normal(val.real, max(1e-6, 1 / shots))
                + 1j * rng.normal(val.imag, max(1e-6, 1 / shots))
                for val in row
            ]
            noisy.append(noisy_row)
        return noisy


__all__ = ["GraphQNN"]
