import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple, Sequence as Seq

import networkx as nx
import numpy as np
import pennylane as qml
import pennylane.numpy as pnp

Tensor = np.ndarray

class GraphQNNGen241:
    """Quantum GraphQNN class using Pennylane.

    Provides quantum feed‑forward, fidelity‑based graph construction,
    and a simple variational training loop.
    """

    @staticmethod
    def random_unitary(num_qubits: int) -> np.ndarray:
        """Generate a random unitary matrix on `num_qubits` qubits."""
        dim = 2 ** num_qubits
        mat = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
        q, _ = np.linalg.qr(mat)
        return q

    @staticmethod
    def random_training_data(unitary: np.ndarray, samples: int) -> List[Tuple[Tensor, Tensor]]:
        """Generate random input state and target state pairs."""
        dataset: List[Tuple[Tensor, Tensor]] = []
        num_qubits = int(np.log2(unitary.shape[0]))
        for _ in range(samples):
            state = np.random.randn(2 ** num_qubits) + 1j * np.random.randn(2 ** num_qubits)
            state = state / np.linalg.norm(state)
            target = unitary @ state
            dataset.append((state, target))
        return dataset

    @staticmethod
    def random_network(qnn_arch: Sequence[int], samples: int):
        """Create a list of variational QNodes for each layer."""
        num_layers = len(qnn_arch) - 1
        dev = qml.device("default.qubit", wires=max(qnn_arch))

        unitaries: List[Tuple[qml.QNode, np.ndarray]] = []

        for layer in range(num_layers):
            in_qubits = qnn_arch[layer]
            out_qubits = qnn_arch[layer + 1]
            params_shape = (out_qubits, 3)  # 3 rotation params per qubit
            params = pnp.random.randn(*params_shape)

            @qml.qnode(dev, interface="autograd")
            def layer_circuit(x: Tensor, params: Tensor):
                # Encode classical input as computational basis
                for i in range(in_qubits):
                    if x[i] < 0:
                        qml.PauliX(i)
                # Apply rotations
                for q in range(out_qubits):
                    qml.RX(params[q, 0], wires=q)
                    qml.RY(params[q, 1], wires=q)
                    qml.RZ(params[q, 2], wires=q)
                # Entanglement
                for q in range(out_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])
                return qml.state()

            unitaries.append((layer_circuit, params))

        target_unitary = GraphQNNGen241.random_unitary(qnn_arch[-1])
        training_data = GraphQNNGen241.random_training_data(target_unitary, samples)
        return list(qnn_arch), unitaries, training_data, target_unitary

    @staticmethod
    def feedforward(
        qnn_arch: Sequence[int],
        unitaries: List[Tuple[qml.QNode, np.ndarray]],
        samples: Iterable[Tuple[Tensor, Tensor]],
    ) -> List[List[Tensor]]:
        """Apply each variational layer sequentially to the input state."""
        all_states: List[List[Tensor]] = []
        for inp, _ in samples:
            states = [inp]
            current = inp
            for layer_circuit, params in unitaries:
                current = layer_circuit(current, params)
                states.append(current)
            all_states.append(states)
        return all_states

    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        """Overlap squared between two pure states."""
        return np.abs(np.vdot(a, b)) ** 2

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[Tensor],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Build weighted adjacency graph from quantum fidelities."""
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNNGen241.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    @staticmethod
    def quantum_kernel(states: Sequence[Tensor]) -> np.ndarray:
        """Return kernel matrix from pairwise fidelities."""
        n = len(states)
        K = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i, n):
                fid = GraphQNNGen241.state_fidelity(states[i], states[j])
                K[i, j] = fid
                K[j, i] = fid
        return K
