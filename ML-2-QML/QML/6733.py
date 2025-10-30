import pennylane as qml
import numpy as np
import networkx as nx
import itertools
from typing import List, Tuple, Sequence, Iterable

State = np.ndarray

def _random_unitary(dim: int) -> np.ndarray:
    X = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    Q, R = np.linalg.qr(X)
    d = np.diag(R)
    ph = d / np.abs(d)
    return Q * ph

def random_training_data(target_unitary: np.ndarray, samples: int) -> List[Tuple[State, State]]:
    n = int(np.log2(target_unitary.shape[0]))
    data = []
    for _ in range(samples):
        vec = np.random.randn(2**n) + 1j * np.random.randn(2**n)
        vec /= np.linalg.norm(vec)
        data.append((vec, target_unitary @ vec))
    return data

def random_network(qnn_arch: Sequence[int], samples: int):
    target_unitary = _random_unitary(2**qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)
    params = [np.random.uniform(0, 2*np.pi, (arch, 3)) for arch in qnn_arch[1:]]
    return list(qnn_arch), params, training_data, target_unitary

def feedforward(qnn_arch: Sequence[int], params: Sequence[np.ndarray], samples: Iterable[Tuple[State, State]]):
    outputs = []
    for x, _ in samples:
        dev = qml.device("default.qubit", wires=len(x))
        @qml.qnode(dev, interface='numpy')
        def circuit(inp):
            qml.StatePrep(inp, wires=range(len(inp)))
            for layer, arch in enumerate(qnn_arch[1:]):
                for q in range(arch):
                    qml.RX(params[layer][q,0], wires=q)
                    qml.RY(params[layer][q,1], wires=q)
                    qml.RZ(params[layer][q,2], wires=q)
                for q in range(arch-1):
                    qml.CNOT(wires=[q, q+1])
            return qml.state()
        state = circuit(x)
        outputs.append([x, state])
    return outputs

def state_fidelity(a: State, b: State) -> float:
    return np.abs(np.vdot(a, b))**2

def fidelity_adjacency(states: Sequence[State], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5):
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(s_i, s_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

class GraphQNN:
    def __init__(self, qnn_arch: Sequence[int], dev: qml.Device | None = None):
        self.qnn_arch = list(qnn_arch)
        self.num_layers = len(qnn_arch)-1
        self.dev = dev or qml.device("default.qubit", wires=max(qnn_arch))
        self.params = [np.random.uniform(0, 2*np.pi, (arch, 3)) for arch in qnn_arch[1:]]
        self.circuit = self._build_circuit()

    def _build_circuit(self):
        @qml.qnode(self.dev, interface='torch')
        def circuit(x):
            qml.StatePrep(x, wires=range(len(x)))
            for layer, arch in enumerate(self.qnn_arch[1:]):
                for q in range(arch):
                    qml.RX(self.params[layer][q,0], wires=q)
                    qml.RY(self.params[layer][q,1], wires=q)
                    qml.RZ(self.params[layer][q,2], wires=q)
                for q in range(arch-1):
                    qml.CNOT(wires=[q, q+1])
            return qml.state()
        return circuit

    def forward(self, x: State) -> List[State]:
        return [self.circuit(x)]

    def train_network(self, dataset: List[Tuple[State, State]], lr: float = 0.01, epochs: int = 100):
        opt = qml.AdamOptimizer(stepsize=lr)
        loss_hist = []
        for _ in range(epochs):
            loss = 0.0
            for inp, target in dataset:
                def loss_fn(params):
                    self.params = params
                    out = self.circuit(inp)
                    return 1 - np.abs(np.vdot(out, target))**2
                self.params = opt.step(loss_fn, self.params)
                loss += loss_fn(self.params)
            loss_hist.append(loss / len(dataset))
        return loss_hist

    def inference(self, x: State) -> State:
        return self.circuit(x)

    @staticmethod
    def from_random(qnn_arch: Sequence[int], samples: int):
        arch, params, training_data, target = random_network(qnn_arch, samples)
        model = GraphQNN(arch)
        model.params = params
        return model, training_data
