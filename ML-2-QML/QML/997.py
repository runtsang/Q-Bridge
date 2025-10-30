import pennylane as qml
import pennylane.numpy as np
import networkx as nx
import itertools
from typing import Iterable, Sequence, Tuple, List

Tensor = np.ndarray

class GraphQNN__gen291:
    """
    Quantum graph neural network that extends the original GraphQNN.
    Implements a parameter‑driven variational circuit using PennyLane,
    with layer‑wise state propagation and fidelity‑based adjacency.
    """

    def __init__(self, qnn_arch: Sequence[int], device_name: str = "default.qubit"):
        self.qnn_arch = list(qnn_arch)
        self.device = qml.device(device_name, wires=self.qnn_arch[-1])
        self.layers = self._build_layers()

    def _build_layers(self):
        layers = []
        for layer in range(1, len(self.qnn_arch)):
            num_inputs = self.qnn_arch[layer - 1]
            num_outputs = self.qnn_arch[layer]
            layer_ops = []
            for _ in range(num_outputs):
                layer_ops.append(self._random_qubit_unitary(num_inputs + 1))
            layers.append(layer_ops)
        return layers

    @staticmethod
    def _random_qubit_unitary(num_qubits: int):
        dev = qml.device("default.qubit", wires=num_qubits)

        @qml.qnode(dev, interface="autograd")
        def circuit(params):
            for w in range(num_qubits):
                qml.RX(params[w, 0], wires=w)
                qml.RY(params[w, 1], wires=w)
                qml.RZ(params[w, 2], wires=w)
            for w in range(num_qubits - 1):
                qml.CNOT(wires=[w, w + 1])
            return qml.state()
        return circuit

    @staticmethod
    def random_training_data(unitary: qml.QNode, samples: int) -> List[Tuple[Tensor, Tensor]]:
        dataset = []
        num_qubits = unitary.num_wires
        for _ in range(samples):
            amp = np.random.normal(size=(2 ** num_qubits, 1))
            amp = amp / np.linalg.norm(amp)
            state = qml.StateVector(amp, wires=range(num_qubits))
            target = unitary(state)
            dataset.append((state, target))
        return dataset

    @staticmethod
    def random_network(qnn_arch: List[int], samples: int):
        target_unitary = GraphQNN__gen291._random_qubit_unitary(qnn_arch[-1])
        training_data = GraphQNN__gen291.random_training_data(target_unitary, samples)

        layers = []
        for layer in range(1, len(qnn_arch)):
            num_inputs = qnn_arch[layer - 1]
            num_outputs = qnn_arch[layer]
            layer_ops = []
            for _ in range(num_outputs):
                layer_ops.append(GraphQNN__gen291._random_qubit_unitary(num_inputs + 1))
            layers.append(layer_ops)

        return qnn_arch, layers, training_data, target_unitary

    @staticmethod
    def feedforward(
        qnn_arch: Sequence[int],
        layers: Sequence[Sequence[qml.QNode]],
        samples: Iterable[Tuple[Tensor, Tensor]],
    ) -> List[List[Tensor]]:
        stored_states = []
        for sample, _ in samples:
            current = sample
            layerwise = [current]
            for layer in range(1, len(qnn_arch)):
                current = layers[layer - 1][0](current)
                layerwise.append(current)
            stored_states.append(layerwise)
        return stored_states

    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        return np.abs(np.vdot(a, b)) ** 2

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[Tensor],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNN__gen291.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    def predict(self, graph: nx.Graph) -> Tensor:
        """
        Predict a vector of fidelities for the given graph.
        The graph is interpreted as an adjacency matrix that defines the
        initial state.  The prediction is the mean fidelity of the
        final layer state with respect to the target unitary.
        """
        adjacency = nx.to_numpy_array(graph, dtype=np.float64)
        state = qml.StateVector(adjacency.flatten(), wires=self.qnn_arch[-1])
        final_state = self.layers[-1][0](state)
        return final_state

    def fit(
        self,
        graphs: Iterable[nx.Graph],
        targets: Iterable[Tensor],
        epochs: int = 100,
        lr: float = 1e-3,
    ) -> None:
        """
        Train the variational parameters to minimise the MSE between
        predicted fidelities and the provided targets.
        """
        opt = qml.GradientDescentOptimizer(lr)
        params = [np.random.uniform(-np.pi, np.pi, size=layer[0].num_params)
                  for layer in self.layers]
        for epoch in range(epochs):
            loss = 0.0
            for g, y in zip(graphs, targets):
                adjacency = nx.to_numpy_array(g, dtype=np.float64)
                state = qml.StateVector(adjacency.flatten(), wires=self.qnn_arch[-1])
                def cost(p):
                    current = state
                    for layer, layer_ops in zip(self.layers, p):
                        current = layer_ops[0](current)
                    return np.mean((current - y) ** 2)
                params = opt.step(cost, params)
                loss += cost(params)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: loss={loss / len(graphs):.4f}")
