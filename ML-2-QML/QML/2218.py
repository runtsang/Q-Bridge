"""Quantum‑centric hybrid graph neural network.

This module implements the same public interface as the classical
GraphQNNHybrid but uses Qiskit state vectors and quantum circuits.
The network can be trained with a parameter‑shaped quantum circuit and
the resulting states can be compared via fidelity to build a graph.
"""

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.neural_networks import SamplerQNN as QSamplerQNN
from qiskit.primitives import Sampler as QSampler


class GraphQNNHybrid:
    """Hybrid graph neural network that operates with quantum circuits."""

    def __init__(self, arch: Sequence[int]) -> None:
        """Create a network with the given layer sizes.

        Parameters
        ----------
        arch:
            Layer sizes, e.g. ``[2, 4, 2]``.
        """
        self.arch = list(arch)

    @staticmethod
    def _random_qubit_unitary(num_qubits: int) -> np.ndarray:
        """Return a random unitary matrix of size 2**num_qubits."""
        dim = 2**num_qubits
        mat = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
        q, _ = np.linalg.qr(mat)
        return q

    @staticmethod
    def _random_qubit_state(num_qubits: int) -> Statevector:
        """Return a random pure state."""
        dim = 2**num_qubits
        vec = np.random.randn(dim) + 1j * np.random.randn(dim)
        vec /= np.linalg.norm(vec)
        return Statevector(vec)

    def random_weights(self) -> List[List[np.ndarray]]:
        """Return a nested list of random unitary matrices for each layer."""
        weights: List[List[np.ndarray]] = [[]]
        for layer in range(1, len(self.arch)):
            num_inputs = self.arch[layer - 1]
            num_outputs = self.arch[layer]
            layer_ops: List[np.ndarray] = []
            for _ in range(num_outputs):
                op = self._random_qubit_unitary(num_inputs + 1)
                layer_ops.append(op)
            weights.append(layer_ops)
        return weights

    def random_training_data(
        self, unitary: Statevector, samples: int
    ) -> List[Tuple[Statevector, Statevector]]:
        """Generate dataset of input–output state pairs."""
        data: List[Tuple[Statevector, Statevector]] = []
        num_qubits = unitary.num_qubits
        for _ in range(samples):
            state = self._random_qubit_state(num_qubits)
            data.append((state, unitary @ state))
        return data

    def feedforward(
        self,
        weights: Sequence[Sequence[np.ndarray]],
        samples: Iterable[Tuple[Statevector, Statevector]],
    ) -> List[List[Statevector]]:
        """Propagate quantum states through the network."""
        outputs: List[List[Statevector]] = []
        for sample, _ in samples:
            layerwise = [sample]
            current = sample
            for layer in range(1, len(self.arch)):
                unitary = weights[layer][0]
                for gate in weights[layer][1:]:
                    unitary = gate @ unitary
                current = Statevector(unitary @ current.data)
                layerwise.append(current)
            outputs.append(layerwise)
        return outputs

    @staticmethod
    def state_fidelity(a: Statevector, b: Statevector) -> float:
        """Return the squared overlap of two pure states."""
        return abs((a.dag() @ b)[0, 0]) ** 2

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[Statevector],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Build a weighted graph from quantum state fidelities."""
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNNHybrid.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    # ------------------------------------------------------------------
    #  Quantum sampler integration
    # ------------------------------------------------------------------
    @staticmethod
    def SamplerQNN() -> QSamplerQNN:
        """Return a Qiskit SamplerQNN circuit with 2‑qubit input and 4‑qubit weight."""
        inputs = ParameterVector("input", 2)
        weights = ParameterVector("weight", 4)

        qc = QuantumCircuit(2)
        qc.ry(inputs[0], 0)
        qc.ry(inputs[1], 1)
        qc.cx(0, 1)
        qc.ry(weights[0], 0)
        qc.ry(weights[1], 1)
        qc.cx(0, 1)
        qc.ry(weights[2], 0)
        qc.ry(weights[3], 1)

        sampler = QSampler()
        return QSamplerQNN(circuit=qc,
                           input_params=inputs,
                           weight_params=weights,
                           sampler=sampler)
