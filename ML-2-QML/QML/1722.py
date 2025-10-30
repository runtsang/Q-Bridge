import numpy as np
import networkx as nx
import itertools
from typing import Iterable, List, Tuple, Sequence
from qiskit import QuantumCircuit, Aer, execute, transpile
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector, Operator
import torch

def _random_unitary(num_qubits: int) -> np.ndarray:
    dim = 2 ** num_qubits
    random_matrix = (np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim))
    q, _ = np.linalg.qr(random_matrix)
    return q

def _random_state(num_qubits: int) -> np.ndarray:
    dim = 2 ** num_qubits
    vec = (np.random.randn(dim) + 1j * np.random.randn(dim))
    vec /= np.linalg.norm(vec)
    return vec

def random_training_data(unitary: np.ndarray, samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    data = []
    num_qubits = int(np.log2(unitary.shape[0]))
    for _ in range(samples):
        state = _random_state(num_qubits)
        target = unitary @ state
        data.append((state, target))
    return data

def random_network(qnn_arch: List[int], samples: int):
    target_unitary = _random_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)

    circuits: List[List[QuantumCircuit]] = [[]]
    for layer in range(1, len(qnn_arch)):
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        layer_ops: List[QuantumCircuit] = []
        for output in range(num_outputs):
            qc = QuantumCircuit(num_inputs + 1)
            for q in range(num_inputs + 1):
                qc.rx(np.random.rand() * 2 * np.pi, q)
                qc.ry(np.random.rand() * 2 * np.pi, q)
                qc.rz(np.random.rand() * 2 * np.pi, q)
            if num_outputs > 1:
                qc.swap(num_inputs, num_inputs + output)
            layer_ops.append(qc)
        circuits.append(layer_ops)

    return qnn_arch, circuits, training_data, target_unitary

def _partial_trace(state: np.ndarray, keep: List[int]) -> np.ndarray:
    full_state = Statevector(state)
    return full_state.partial_trace(keep).data

def _layer_channel(qnn_arch: Sequence[int], circuits: Sequence[List[QuantumCircuit]], layer: int, input_state: np.ndarray) -> np.ndarray:
    num_inputs = qnn_arch[layer - 1]
    state = Statevector(input_state)
    for qc in circuits[layer]:
        unitary = Operator(qc).data
        state = state.evolve(unitary)
    keep = list(range(num_inputs))
    return _partial_trace(state.data, keep)

def feedforward(qnn_arch: Sequence[int], circuits: Sequence[List[QuantumCircuit]], samples: Iterable[Tuple[np.ndarray, np.ndarray]]) -> List[List[np.ndarray]]:
    stored = []
    for sample, _ in samples:
        layerwise = [sample]
        current = sample
        for layer in range(1, len(qnn_arch)):
            current = _layer_channel(qnn_arch, circuits, layer, current)
            layerwise.append(current)
        stored.append(layerwise)
    return stored

def state_fidelity(a: np.ndarray, b: np.ndarray) -> float:
    return abs(np.vdot(a, b)) ** 2

def fidelity_adjacency(states: Sequence[np.ndarray], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

class GraphQNN:
    """
    Quantum graph neural network that learns a unitary mapping from input qubits to output qubits via a variational circuit.
    """
    def __init__(self, qnn_arch: List[int], backend_name: str = "aer_simulator", shots: int = 1024):
        self.qnn_arch = qnn_arch
        self.backend = Aer.get_backend(backend_name)
        self.shots = shots
        self.circuits: List[List[QuantumCircuit]] = [[]]
        self.params: List[Parameter] = []

    def _build_circuit(self, layer: int, num_inputs: int, num_outputs: int) -> QuantumCircuit:
        qc = QuantumCircuit(num_inputs + 1)
        for q in range(num_inputs + 1):
            θ = Parameter(f"θ_{layer}_{q}")
            self.params.append(θ)
            qc.rx(θ, q)
        if num_outputs > 1:
            qc.swap(num_inputs, num_inputs + (num_outputs - 1))
        return qc

    def initialize(self):
        for layer in range(1, len(self.qnn_arch)):
            num_inputs = self.qnn_arch[layer - 1]
            num_outputs = self.qnn_arch[layer]
            layer_ops = [self._build_circuit(layer, num_inputs, num_outputs) for _ in range(num_outputs)]
            self.circuits.append(layer_ops)

    def evaluate(self, state: np.ndarray, param_values: List[float]) -> np.ndarray:
        bound_circuits = []
        for layer, ops in enumerate(self.circuits[1:], start=1):
            for op in ops:
                bound_circuits.append(op.bind_parameters({p: v for p, v in zip(self.params, param_values)}))
        current = Statevector(state)
        for qc in bound_circuits:
            unitary = Operator(qc).data
            current = current.evolve(unitary)
        return current.data

    def train(self, training_data: List[Tuple[np.ndarray, np.ndarray]], epochs: int = 20, lr: float = 0.01):
        opt = torch.optim.Adam([torch.tensor([0.0], requires_grad=True)], lr=lr)
        for epoch in range(epochs):
            total_loss = 0.0
            for inp, tgt in training_data:
                pred = self.evaluate(inp, opt.param_groups[0]['params'])
                loss = torch.mean((torch.tensor(pred.real) - torch.tensor(tgt.real)) ** 2)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item()
            # print(f"Epoch {epoch+1}/{epochs} Loss: {total_loss/len(training_data):.4f}")

    def fidelity_adj(self, states: List[np.ndarray], threshold: float) -> nx.Graph:
        return fidelity_adjacency(states, threshold)

__all__ = [
    "GraphQNN",
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
]
