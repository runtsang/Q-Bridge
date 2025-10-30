import numpy as np
import networkx as nx
import itertools
from typing import List, Tuple, Sequence, Iterable
import pennylane as qml

class GraphQNNGen192:
    """Quantum graph‑based neural network utilities with PennyLane and fidelity‑based adjacency."""

    @staticmethod
    def _random_qubit_state(num_qubits: int, rng: np.random.Generator) -> np.ndarray:
        dim = 2 ** num_qubits
        state = rng.normal(size=(dim,)) + 1j * rng.normal(size=(dim,))
        state /= np.linalg.norm(state)
        return state

    @staticmethod
    def _random_qubit_unitary(num_qubits: int, rng: np.random.Generator) -> np.ndarray:
        dim = 2 ** num_qubits
        mat = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
        q, _ = np.linalg.qr(mat)
        return q

    @staticmethod
    def _layer_unitary(num_qubits: int, params: np.ndarray) -> np.ndarray:
        dev = qml.device("default.qubit", wires=num_qubits)

        @qml.qnode(dev, interface="numpy")
        def circuit(params):
            for qubit in range(num_qubits):
                qml.Rot(params[qubit, 0], params[qubit, 1], params[qubit, 2], wires=qubit)
            for qubit in range(num_qubits - 1):
                qml.CNOT(wires=[qubit, qubit + 1])
            return qml.state()

        return qml.matrix(circuit, argnums=0)(params)

    @staticmethod
    def random_training_data(unitary: np.ndarray, samples: int, rng: np.random.Generator) -> List[Tuple[np.ndarray, np.ndarray]]:
        dataset: List[Tuple[np.ndarray, np.ndarray]] = []
        dim = unitary.shape[0]
        for _ in range(samples):
            state = GraphQNNGen192._random_qubit_state(int(np.log2(dim)), rng)
            target = unitary @ state
            dataset.append((state, target))
        return dataset

    @staticmethod
    def random_network(qnn_arch: Sequence[int], samples: int, rng: np.random.Generator) -> Tuple[List[int], List[List[np.ndarray]], List[Tuple[np.ndarray, np.ndarray]], np.ndarray]:
        params: List[List[np.ndarray]] = [[]]
        for layer in range(1, len(qnn_arch)):
            num_qubits = qnn_arch[layer - 1]
            layer_params = rng.normal(size=(num_qubits, 3))
            params.append([layer_params])
        target_unitary = GraphQNNGen192._random_qubit_unitary(qnn_arch[-1], rng)
        training_data = GraphQNNGen192.random_training_data(target_unitary, samples, rng)
        return list(qnn_arch), params, training_data, target_unitary

    @staticmethod
    def feedforward(qnn_arch: Sequence[int], params: Sequence[Sequence[np.ndarray]], samples: Iterable[Tuple[np.ndarray, np.ndarray]]) -> List[List[np.ndarray]]:
        states_per_sample: List[List[np.ndarray]] = []
        for input_state, _ in samples:
            layerwise: List[np.ndarray] = [input_state]
            current = input_state
            for layer_idx in range(1, len(qnn_arch)):
                layer_params = params[layer_idx][0]
                unitary = GraphQNNGen192._layer_unitary(qnn_arch[layer_idx - 1], layer_params)
                current = unitary @ current
                layerwise.append(current)
            states_per_sample.append(layerwise)
        return states_per_sample

    @staticmethod
    def state_fidelity(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.abs(np.vdot(a, b)) ** 2)

    @staticmethod
    def fidelity_adjacency(states: Sequence[np.ndarray], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNNGen192.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    @staticmethod
    def _flatten_params(params: List[List[np.ndarray]]) -> np.ndarray:
        flat_list: List[np.ndarray] = []
        for layer in params:
            for arr in layer:
                flat_list.append(arr.ravel())
        return np.concatenate(flat_list)

    @staticmethod
    def _unflatten_params(flat: np.ndarray, arch: List[int]) -> List[List[np.ndarray]]:
        params: List[List[np.ndarray]] = [[]]
        idx = 0
        for layer in range(1, len(arch)):
            num_qubits = arch[layer - 1]
            size = num_qubits * 3
            arr = flat[idx:idx + size].reshape((num_qubits, 3))
            idx += size
            params.append([arr])
        return params

    @staticmethod
    def _cost(flat_params: np.ndarray, arch: List[int], training_data: List[Tuple[np.ndarray, np.ndarray]]) -> float:
        params = GraphQNNGen192._unflatten_params(flat_params, arch)
        total_loss = 0.0
        for input_state, target_state in training_data:
            current = input_state
            for layer_idx in range(1, len(arch)):
                layer_params = params[layer_idx][0]
                unitary = GraphQNNGen192._layer_unitary(arch[layer_idx - 1], layer_params)
                current = unitary @ current
            loss = np.mean(np.abs(current - target_state) ** 2)
            total_loss += loss
        return total_loss / len(training_data)

    @staticmethod
    def train(arch: List[int], params: List[List[np.ndarray]], training_data: List[Tuple[np.ndarray, np.ndarray]],
              lr: float = 0.01, epochs: int = 100) -> List[List[np.ndarray]]:
        flat_params = GraphQNNGen192._flatten_params(params)
        for _ in range(epochs):
            loss = GraphQNNGen192._cost(flat_params, arch, training_data)
            grad_fn = qml.gradients.param_shift(lambda fp: GraphQNNGen192._cost(fp, arch, training_data), 0)
            grad = grad_fn(flat_params)
            flat_params -= lr * grad
        updated_params = GraphQNNGen192._unflatten_params(flat_params, arch)
        return updated_params

__all__ = ["GraphQNNGen192"]
