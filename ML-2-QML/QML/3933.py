"""Quantum implementation of the integrated graph‑quanvolutional network.

The module mirrors the classical counterpart but uses variational
unitaries for the graph layers and a quantum kernel for the front‑end.
"""
import torch
import torchquantum as tq
import networkx as nx
import itertools
import scipy as sc
import numpy as np
from typing import List, Tuple, Sequence, Iterable

def _random_qubit_unitary(num_qubits: int) -> tq.Qobj:
    """Generate a random unitary on `num_qubits` qubits."""
    dim = 2 ** num_qubits
    matrix = sc.random.normal(size=(dim, dim)) + 1j * sc.random.normal(size=(dim, dim))
    unitary = sc.linalg.orth(matrix)
    return tq.Qobj(unitary)

def random_training_data(unitary: tq.Qobj, samples: int) -> List[Tuple[tq.Qobj, tq.Qobj]]:
    """Create synthetic training pairs for a unitary layer."""
    dataset: List[Tuple[tq.Qobj, tq.Qobj]] = []
    num_qubits = int(np.log2(unitary.shape[0]))
    for _ in range(samples):
        state = tq.random_state(num_qubits)
        dataset.append((state, unitary * state))
    return dataset

def random_network(qnn_arch: List[int], samples: int):
    """Generate a random variational graph network."""
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)
    unitaries: List[List[tq.Qobj]] = [[]]
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        layer_ops: List[tq.Qobj] = []
        for _ in range(out_f):
            op = _random_qubit_unitary(in_f + 1)
            layer_ops.append(op)
        unitaries.append(layer_ops)
    return qnn_arch, unitaries, training_data, target_unitary

def _partial_trace(state: tq.Qobj, keep: Sequence[int]) -> tq.Qobj:
    """Partial trace over all but the qubits in `keep`."""
    return state.ptrace(list(keep))

def _layer_channel(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[tq.Qobj]], layer: int, input_state: tq.Qobj) -> tq.Qobj:
    """Apply the variational layer at index `layer`."""
    num_inputs = qnn_arch[layer - 1]
    num_outputs = qnn_arch[layer]
    # Prepare extended state with auxiliary zero qubits
    state = tq.tensor(input_state, tq.zero_state(num_outputs))
    # Combine all gates of the layer
    layer_unitary = unitaries[layer][0]
    for gate in unitaries[layer][1:]:
        layer_unitary = gate * layer_unitary
    evolved = layer_unitary * state * layer_unitary.dag()
    return _partial_trace(evolved, list(range(num_inputs)))

def feedforward(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[tq.Qobj]], samples: Iterable[Tuple[tq.Qobj, tq.Qobj]]):
    """Forward pass through the quantum graph network."""
    stored_states = []
    for sample, _ in samples:
        layerwise = [sample]
        current = sample
        for layer in range(1, len(qnn_arch)):
            current = _layer_channel(qnn_arch, unitaries, layer, current)
            layerwise.append(current)
        stored_states.append(layerwise)
    return stored_states

def state_fidelity(a: tq.Qobj, b: tq.Qobj) -> float:
    """Squared overlap between two pure states."""
    return abs((a.dag() * b)[0, 0]) ** 2

def fidelity_adjacency(states: Sequence[tq.Qobj], threshold: float,
                       *, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
    """Create a weighted adjacency graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(s_i, s_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

class QuantumQuanvolutionFilter(tq.QuantumModule):
    """Quantum kernel applied to 2×2 image patches."""
    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(n_wires)]
        )
        self.random_layer = tq.RandomLayer(n_ops=8, wires=list(range(n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.encoder.num_wires, bsz=bsz, device=device)
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                data = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, data)
                self.random_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)

class IntegratedGraphQuanvolutionQNN(tq.QuantumModule):
    """Quantum graph‑quanvolution neural network."""
    def __init__(self, qnn_arch: Sequence[int], n_wires: int = 4):
        super().__init__()
        self.qnn_arch = list(qnn_arch)
        self.qfilter = QuantumQuanvolutionFilter(n_wires)
        _, self.unitaries, _, _ = random_network(self.qnn_arch, samples=10)

    def forward(self, x: torch.Tensor) -> List[tq.Qobj]:
        """Apply the quantum quanvolution filter and then the graph layers."""
        # Quantum filter outputs classical measurement vector
        qfeatures = self.qfilter(x)
        # Convert each measurement vector into a state
        states = []
        for vec in qfeatures:
            vec_np = vec.cpu().numpy()
            norm = np.linalg.norm(vec_np) + 1e-12
            state = tq.Qobj(vec_np.reshape(-1, 1) / norm)
            states.append(state)
        # Process each state through the quantum graph
        final_states: List[tq.Qobj] = []
        for state in states:
            current = state
            for layer in range(1, len(self.qnn_arch)):
                current = _layer_channel(self.qnn_arch, self.unitaries, layer, current)
            final_states.append(current)
        return final_states

    def get_fidelity_graph(self, states: Sequence[tq.Qobj], threshold: float,
                           *, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
        return fidelity_adjacency(states, threshold, secondary=secondary, secondary_weight=secondary_weight)

__all__ = [
    "IntegratedGraphQuanvolutionQNN",
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "QuantumQuanvolutionFilter",
]
