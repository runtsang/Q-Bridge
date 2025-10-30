"""
Quantum graph neural network utilities based on PennyLane.

This module implements a variational quantum circuit that mimics a
classical GNN layer.  Each node is represented by a qubit; layers
consist of parameterised single‑qubit rotations followed by a
controlled‑entanglement pattern that implements a graph convolution.
The public API mirrors the seed functions but returns a `GraphQNN`
class that can generate random parameter sets, perform a forward
pass, and compute fidelity‑based adjacency graphs.
"""

import pennylane as qml
import numpy as np
import networkx as nx
import itertools
from typing import Iterable, Sequence, List, Tuple

def _random_qubit_unitary(num_qubits: int) -> np.ndarray:
    """Return a random unitary matrix for *num_qubits* qubits."""
    dim = 2 ** num_qubits
    random_matrix = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    q, _ = np.linalg.qr(random_matrix)
    return q

def _random_qubit_state(num_qubits: int) -> np.ndarray:
    """Sample a random pure state on *num_qubits* qubits."""
    dim = 2 ** num_qubits
    state = np.random.randn(dim) + 1j * np.random.randn(dim)
    state /= np.linalg.norm(state)
    return state

def random_training_data(unitary: np.ndarray, samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate synthetic training data using a target unitary."""
    dataset: List[Tuple[np.ndarray, np.ndarray]] = []
    num_qubits = int(np.log2(unitary.shape[0]))
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        dataset.append((state, unitary @ state))
    return dataset

class GraphQNN:
    """
    Variational quantum graph neural network.

    Parameters
    ----------
    arch : Sequence[int]
        Layer widths.  The first element is the number of input qubits.
    dev : pennylane.Device
        PennyLane device used to evaluate circuits.
    """

    def __init__(self, arch: Sequence[int], dev: qml.Device):
        self.arch = list(arch)
        self.dev = dev
        # Parameters are stored as a list of numpy arrays, one per layer
        self.params: List[np.ndarray] = []
        for layer in range(1, len(self.arch)):
            num_outputs = self.arch[layer]
            # Each output qubit receives a set of rotation angles (RY, RZ, RX)
            layer_params = np.random.randn(num_outputs, 3)
            self.params.append(layer_params)

    @classmethod
    def random_network(cls, arch: Sequence[int], samples: int):
        """Instantiate a network with random parameters and synthetic training data."""
        dev = qml.device("default.qubit", wires=max(arch))
        net = cls(arch, dev)
        target_unitary = _random_qubit_unitary(arch[-1])
        training_data = random_training_data(target_unitary, samples)
        return net, training_data, target_unitary

    def _layer(self, qubits: List[int], params: np.ndarray):
        """Apply a single GNN layer consisting of rotations and entanglement."""
        for q, (ry, rz, rx) in zip(qubits, params):
            qml.RY(ry, wires=q)
            qml.RZ(rz, wires=q)
            qml.RX(rx, wires=q)

        # Simple entanglement pattern: controlled‑NOT between consecutive qubits
        for i in range(len(qubits) - 1):
            qml.CNOT(wires=[qubits[i], qubits[i + 1]])

    def circuit(self, input_state: np.ndarray, params: List[np.ndarray]) -> List[np.ndarray]:
        """Execute the full circuit and return state vectors per layer."""
        @qml.qnode(self.dev, interface="np")
        def circuit_fn(state):
            qml.QubitStateVector(state, wires=range(self.arch[0]))
            states = [state]
            for layer_params in params:
                self._layer(range(self.arch[0]), layer_params)
                states.append(qml.state())
            return states

        return circuit_fn(input_state)

    def feedforward(self,
                    samples: Iterable[Tuple[np.ndarray, np.ndarray]]) -> List[List[np.ndarray]]:
        """Run a forward pass and capture state vectors per layer."""
        stored: List[List[np.ndarray]] = []
        for state, _ in samples:
            layerwise = self.circuit(state, self.params)
            stored.append(layerwise)
        return stored

    @staticmethod
    def state_fidelity(a: np.ndarray, b: np.ndarray) -> float:
        """Squared absolute overlap between two pure states."""
        return abs(np.vdot(a, b)) ** 2

    @staticmethod
    def fidelity_adjacency(states: Sequence[np.ndarray],
                           threshold: float,
                           *,
                           secondary: float | None = None,
                           secondary_weight: float = 0.5) -> nx.Graph:
        """Build a weighted graph from state fidelities."""
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNN.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    def get_state_vectors(self,
                          samples: Iterable[Tuple[np.ndarray, np.ndarray]]) -> List[np.ndarray]:
        """Return the final layer state vectors for a batch of samples."""
        outputs = self.feedforward(samples)
        return [out[-1] for out in outputs]

    def build_graph_from_states(self,
                                states: List[np.ndarray],
                                threshold: float) -> nx.Graph:
        """Construct an adjacency graph directly from quantum states."""
        return self.fidelity_adjacency(states, threshold)
