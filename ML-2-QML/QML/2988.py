from __future__ import annotations

import itertools
from typing import Iterable, Sequence, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
import networkx as nx

class GraphQNNGen256(tq.QuantumModule):
    """Quantum graph neural network that mirrors the classical GraphQNNGen256.
    It encodes image features into a small qubit register, applies a stack
    of random unitary layers, measures all qubits, and normalises the
    resulting amplitudes."""
    def __init__(
        self,
        arch: Sequence[int] = (256,),
        threshold: float = 0.9,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ):
        super().__init__()
        self.arch = list(arch)
        self.threshold = threshold
        self.secondary = secondary
        self.secondary_weight = secondary_weight

        # Encoder: use a fixed RyZXY pattern on the first four qubits
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        # Random layers per graph layer
        self.layers = nn.ModuleList()
        for layer_idx in range(1, len(arch)):
            wires = list(range(arch[layer_idx - 1] + arch[layer_idx]))
            self.layers.append(tq.RandomLayer(n_ops=50, wires=wires))
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.arch[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.arch[-1], bsz=bsz, device=x.device, record_op=True)
        # Encode image features into the first four qubits
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        # Apply random layers sequentially
        for layer in self.layers:
            layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)

    @staticmethod
    def state_fidelity(a: tq.Qobj, b: tq.Qobj) -> float:
        """Absolute squared overlap between two pure states."""
        return abs((a.dag() * b)[0, 0]) ** 2

    def fidelity_adjacency(self, states: Sequence[tq.Qobj]) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
            fid = self.state_fidelity(s_i, s_j)
            if fid >= self.threshold:
                graph.add_edge(i, j, weight=1.0)
            elif self.secondary is not None and fid >= self.secondary:
                graph.add_edge(i, j, weight=self.secondary_weight)
        return graph

    @staticmethod
    def random_network(arch: Sequence[int], samples: int):
        """Generate a random quantum network consistent with the
        specified architecture."""
        target_unitary = tq.RandomUnitary(arch[-1])
        training_data = [(tq.RandomState(arch[-1]), target_unitary * tq.RandomState(arch[-1])) for _ in range(samples)]
        unitaries: List[List[tq.Qobj]] = [[]]
        for layer in range(1, len(arch)):
            layer_ops = []
            for _ in range(arch[layer]):
                op = tq.RandomUnitary(arch[layer - 1] + 1)
                layer_ops.append(op)
            unitaries.append(layer_ops)
        return arch, unitaries, training_data, target_unitary

    @staticmethod
    def random_training_data(unitary: tq.Qobj, samples: int):
        dataset = []
        for _ in range(samples):
            state = tq.RandomState(unitary.dims[0][0])
            dataset.append((state, unitary * state))
        return dataset

__all__ = ["GraphQNNGen256"]
