import numpy as np
import networkx as nx
import itertools
from typing import Iterable, Sequence, List, Tuple
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.quantum_info import Statevector, random_unitary
from qiskit.providers.aer import AerSimulator

class GraphQNN__gen292:
    """
    Variational quantum graph neural network that mirrors the classical
    architecture.  Each layer is a parameter‑free random unitary acting on
    the concatenation of its input qubits and the qubits that will become
    the layer's outputs.  The circuit depth is controlled by ``depth``.
    """

    def __init__(self, arch: Sequence[int], depth: int = 1):
        self.arch = list(arch)
        self.depth = depth
        # Build a random unitary for each layer
        self.unitaries: List[QuantumCircuit] = []
        for layer in range(1, len(self.arch)):
            num_inputs = self.arch[layer - 1]
            num_outputs = self.arch[layer]
            num_qubits = num_inputs + num_outputs
            circ = QuantumCircuit(num_qubits)
            unitary_matrix = random_unitary(num_qubits).data
            circ.append(unitary_matrix, range(num_qubits))
            self.unitaries.append(circ)

    # ------------------------------------------------------------------
    #  Core helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _partial_trace(state: Statevector, keep: Sequence[int]) -> Statevector:
        """
        Return a new Statevector after tracing out all qubits *not* in ``keep``.
        """
        all_qubits = list(range(state.num_qubits))
        trace_out = [q for q in all_qubits if q not in keep]
        return state.partial_trace(trace_out)

    # ------------------------------------------------------------------
    #  Forward pass
    # ------------------------------------------------------------------
    def feedforward(
        self,
        samples: Iterable[Tuple[Statevector, Statevector]]
    ) -> List[List[Statevector]]:
        """
        Execute the variational circuit on each sample and return the
        state after each layer.

        Parameters
        ----------
        samples : iterable of (input_state, target_state) tuples

        Returns
        -------
        outputs : list[list[Statevector]]
            Layerwise states for each sample.
        """
        simulator = AerSimulator(method="statevector")
        outputs: List[List[Statevector]] = []

        for sample, _ in samples:
            layerwise = [sample]
            current = sample
            for layer_idx, circ in enumerate(self.unitaries):
                # Transpile for the simulator
                transpiled = transpile(circ, simulator)
                current = current.evolve(transpiled)
                # Keep only the output qubits for the next layer
                num_inputs = self.arch[layer_idx]
                num_outputs = self.arch[layer_idx + 1]
                keep = list(range(num_inputs, num_inputs + num_outputs))
                current = self._partial_trace(current, keep)
                layerwise.append(current)
            outputs.append(layerwise)
        return outputs

    # ------------------------------------------------------------------
    #  Data generation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def random_training_data(unitary: Statevector, samples: int) -> List[Tuple[Statevector, Statevector]]:
        """
        Generate a training set where each target is the unitary applied
        to a random input state.

        Parameters
        ----------
        unitary : Statevector
            Target unitary.
        samples : int
            Number of training examples.

        Returns
        -------
        dataset : list[tuple[Statevector, Statevector]]
            (input_state, target_state) pairs.
        """
        dataset: List[Tuple[Statevector, Statevector]] = []
        num_qubits = unitary.num_qubits
        for _ in range(samples):
            input_state = Statevector.random(num_qubits)
            target_state = unitary.evolve(input_state)
            dataset.append((input_state, target_state))
        return dataset

    @staticmethod
    def random_network(arch: List[int], samples: int):
        """
        Build a random variational network and a corresponding training set.

        Parameters
        ----------
        arch : list[int]
            Layer sizes.
        samples : int
            Number of training examples.

        Returns
        -------
        arch : list[int]
            Architecture.
        unitaries : list[QuantumCircuit]
            List of layer circuits.
        training_data : list[tuple[Statevector, Statevector]]
            Training set.
        target_unitary : Statevector
            The unitary that generated the training targets.
        """
        target_unitary = Statevector(random_unitary(arch[-1]).data)
        training_data = GraphQNN__gen292.random_training_data(target_unitary, samples)

        # Build layer circuits
        unitaries: List[QuantumCircuit] = []
        for layer in range(1, len(arch)):
            num_inputs = arch[layer - 1]
            num_outputs = arch[layer]
            num_qubits = num_inputs + num_outputs
            circ = QuantumCircuit(num_qubits)
            circ.append(random_unitary(num_qubits).data, range(num_qubits))
            unitaries.append(circ)

        return arch, unitaries, training_data, target_unitary

    # ------------------------------------------------------------------
    #  Fidelity helpers
    # ------------------------------------------------------------------
    @staticmethod
    def state_fidelity(a: Statevector, b: Statevector) -> float:
        """
        Return the absolute squared overlap between two pure states.
        """
        return abs((a.dag() @ b)[0, 0]) ** 2

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[Statevector],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """
        Create a weighted adjacency graph from state fidelities.

        Parameters
        ----------
        states : sequence of Statevector
            Statevectors.
        threshold : float
            Primary fidelity threshold.
        secondary : float | None
            Secondary threshold for weaker edges.
        secondary_weight : float
            Weight assigned to secondary edges.

        Returns
        -------
        graph : networkx.Graph
            Weighted adjacency graph.
        """
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNN__gen292.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph


__all__ = [
    "GraphQNN__gen292",
]
