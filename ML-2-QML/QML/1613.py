import pennylane as qml
import torch
import torch.nn as nn
import networkx as nx
import itertools
import numpy as np
from typing import Iterable, Sequence, Tuple

def state_fidelity(a: np.ndarray, b: np.ndarray) -> float:
    """Return the squared magnitude of the inner product of two state vectors."""
    return float(np.abs(a.conj().T @ b) ** 2)

def fidelity_adjacency(states: Sequence[np.ndarray], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5):
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

def _build_circuit_from_params(arch: Sequence[int], params: Sequence[torch.Tensor]):
    """Create a PennyLane QNode that implements the variational layers with given parameters."""
    n_wires = arch[-1]
    dev = qml.device("default.qubit", wires=n_wires)

    @qml.qnode(dev, interface="torch")
    def circuit(inputs: torch.Tensor):
        qml.StatePrep(inputs, wires=range(n_wires))
        for idx, (in_f, out_f) in enumerate(zip(arch[:-1], arch[1:])):
            for q_out in range(out_f):
                for q_in in range(in_f):
                    ang = params[idx][q_out, q_in, :].numpy()
                    qml.Rot(*ang, wires=q_out)
            for q_in in range(in_f):
                for q_out in range(out_f):
                    qml.CNOT(wires=[q_in, q_out])
        return qml.state()
    return circuit

def random_training_data(unitary_circuit, samples: int):
    """Generate random input states and corresponding target states using the provided unitary circuit."""
    n_wires = len(unitary_circuit.device.wires)
    dataset: list[Tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(samples):
        basis_index = np.random.randint(0, 2 ** n_wires)
        state = torch.zeros(2 ** n_wires, dtype=torch.complex128)
        state[basis_index] = 1.0
        target = unitary_circuit(state)
        dataset.append((state, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    """Generate architecture, initial parameters, training data, and a target unitary."""
    # Sample target parameters
    target_params = [torch.rand(out, in_, 3) * 2 * np.pi - np.pi
                     for in_, out in zip(qnn_arch[:-1], qnn_arch[1:])]
    target_circuit = _build_circuit_from_params(qnn_arch, target_params)
    training_data = random_training_data(target_circuit, samples)
    # Initial (untrained) parameters
    init_params = [torch.rand(out, in_, 3) * 2 * np.pi - np.pi
                   for in_, out in zip(qnn_arch[:-1], qnn_arch[1:])]
    return list(qnn_arch), init_params, training_data, target_circuit

class GraphQNN:
    """Quantum graph neural network implemented with PennyLane."""

    def __init__(self, arch: Sequence[int], device: str = "default.qubit"):
        self.arch = list(arch)
        self.device = qml.device(device, wires=self.arch[-1])
        self.params = nn.ParameterList()
        for in_f, out_f in zip(arch[:-1], arch[1:]):
            param = nn.Parameter(torch.rand(out_f, in_f, 3) * 2 * np.pi - np.pi)
            self.params.append(param)

    def circuit(self, inputs: torch.Tensor):
        dev = self.device

        @qml.qnode(dev, interface="torch")
        def _circuit():
            qml.StatePrep(inputs, wires=range(self.arch[-1]))
            for idx, (in_f, out_f) in enumerate(zip(self.arch[:-1], self.arch[1:])):
                for q_out in range(out_f):
                    for q_in in range(in_f):
                        ang = self.params[idx][q_out, q_in, :].numpy()
                        qml.Rot(*ang, wires=q_out)
                for q_in in range(in_f):
                    for q_out in range(out_f):
                        qml.CNOT(wires=[q_in, q_out])
            return qml.state()
        return _circuit()

    def forward(self, input_state: torch.Tensor) -> list[torch.Tensor]:
        return [self.circuit(input_state)]

    def train_step(self, data_loader, lr: float = 1e-3, epochs: int = 10):
        optimizer = torch.optim.Adam(self.params, lr=lr)
        for epoch in range(epochs):
            for inp, tgt in data_loader:
                optimizer.zero_grad()
                pred = self.forward(inp)[0]
                loss = torch.mean((pred - tgt) ** 2)
                loss.backward()
                optimizer.step()

def feedforward(qnn_arch: Sequence[int], params: Sequence[torch.Tensor], samples: Iterable[Tuple[torch.Tensor, torch.Tensor]]):
    qnn = GraphQNN(qnn_arch)
    for idx, p in enumerate(params):
        with torch.no_grad():
            qnn.params[idx].copy_(p)
    outputs = []
    for inp, _ in samples:
        out = qnn.forward(inp)[0]
        outputs.append(out)
    return outputs

__all__ = [
    "GraphQNN",
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
]
