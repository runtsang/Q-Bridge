"""GraphQNNGen207: Quantum graph‑neural‑network with optional self‑attention.

This implementation uses Qiskit to construct variational unitaries for each layer, optionally inserting a quantum self‑attention sub‑circuit.  The public API matches the classical version so the two can be swapped in experiments.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector

# Import the quantum self‑attention helper.
from SelfAttention import SelfAttention

# Default simulator backend.
backend = qiskit.Aer.get_backend("qasm_simulator")


class GraphQNNGen207:
    """
    Quantum graph‑neural‑network with optional quantum self‑attention.
    """

    def __init__(self, arch: Sequence[int], use_attention: bool = False):
        """
        Parameters
        ----------
        arch : Sequence[int]
            Architecture list; each entry is the number of qubits in that layer.
        use_attention : bool, optional
            If True, a quantum self‑attention sub‑circuit is appended after each layer.
        """
        self.arch = list(arch)
        self.use_attention = use_attention
        self.unitaries: List[List[QuantumCircuit]] = []

        # Build a random unitary per layer.
        for layer in range(1, len(arch)):
            layer_ops: List[QuantumCircuit] = []
            for _ in range(arch[layer]):
                # Random single‑qubit unitary on the output qubit plus a placeholder for extra qubits.
                qc = QuantumCircuit(arch[layer - 1] + 1)
                qc.h(arch[layer - 1])  # placeholder gate
                qc.rx(np.random.randn(), arch[layer - 1])
                qc.ry(np.random.randn(), arch[layer - 1])
                qc.rz(np.random.randn(), arch[layer - 1])
                layer_ops.append(qc)
            self.unitaries.append(layer_ops)

        if use_attention:
            self.attention = SelfAttention()  # quantum self‑attention class

    @staticmethod
    def random_network(arch: Sequence[int], samples: int, use_attention: bool = False):
        """
        Create a random quantum network and training data.

        Returns
        -------
        arch : List[int]
            Architecture list.
        model : GraphQNNGen207
            Randomly initialized network.
        training_data : List[Tuple[Statevector, Statevector]]
            (input, target) pairs where target is the state after the final layer.
        target_unitary : QuantumCircuit
            The unitary of the last layer (used for data generation).
        """
        model = GraphQNNGen207(arch, use_attention=use_attention)

        # Use the last layer's first unitary as a stand‑in target.
        target_unitary = model.unitaries[-1][0]

        training_data = []
        for _ in range(samples):
            # Start from |0…0> and apply the target unitary to generate a target state.
            state = Statevector.from_label("0" * arch[0])
            target_state = state.evolve(target_unitary)
            training_data.append((state, target_state))
        return arch, model, training_data, target_unitary

    def _apply_layer(self, state: Statevector, layer_idx: int) -> Statevector:
        """
        Apply the unitary for a given layer and optionally the attention block.
        """
        for qc in self.unitaries[layer_idx]:
            state = state.evolve(qc)
        if self.use_attention:
            # Build the attention sub‑circuit for this layer.
            attn_qc = self.attention._build_circuit(
                rotation_params=np.random.randn(3 * self.arch[layer_idx]),
                entangle_params=np.random.randn(self.arch[layer_idx] - 1),
            )
            state = state.evolve(attn_qc)
        return state

    def feedforward(
        self, samples: Iterable[Tuple[Statevector, Statevector]]
    ) -> List[List[Statevector]]:
        """
        Run a batch of input states through the quantum network.

        Parameters
        ----------
        samples : Iterable[Tuple[Statevector, Statevector]]
            Each element is (input, target).  Target is ignored.

        Returns
        -------
        stored : List[List[Statevector]]
            List of state vectors per sample: [s0, s1,..., sn]
        """
        stored: List[List[Statevector]] = []
        for s, _ in samples:
            layerwise = [s]
            current = s
            for layer in range(1, len(self.arch)):
                current = self._apply_layer(current, layer - 1)
                layerwise.append(current)
            stored.append(layerwise)
        return stored

    @staticmethod
    def state_fidelity(a: Statevector, b: Statevector) -> float:
        """
        Return the squared absolute overlap between two pure states.
        """
        return abs(np.vdot(a.data, b.data)) ** 2

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[Statevector],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """
        Build a weighted graph from pairwise state fidelities.
        """
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNNGen207.state_fidelity(a, b)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph


__all__ = [
    "GraphQNNGen207",
]
