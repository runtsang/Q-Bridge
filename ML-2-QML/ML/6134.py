import torch
import numpy as np
import networkx as nx
import itertools
import torchquantum as tq
from torch import nn
from torchquantum.functional import func_name_dict, op_name_dict
from typing import Sequence, Iterable

class HybridKernelGraphQNN:
    """Hybrid classical‑quantum kernel and graph‑neural‑network utilities.

    Combines a scalable classical RBF kernel, a TorchQuantum variational kernel,
    and fidelity‑based graph construction.  The API is intentionally
    minimal to keep the module lightweight yet expressive.
    """

    # -------------------- Classical RBF kernel --------------------
    class _ClassicalRBFKernel(nn.Module):
        def __init__(self, gamma: float = 1.0) -> None:
            super().__init__()
            self.gamma = gamma

        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            x = x.view(-1, x.shape[-1])
            y = y.view(-1, y.shape[-1])
            diff = x[:, None, :] - y[None, :, :]
            d2 = torch.sum(diff * diff, dim=-1)
            return torch.exp(-self.gamma * d2)

    # -------------------- Quantum kernel via TorchQuantum --------------------
    class _QuantumFeatureMap(tq.QuantumModule):
        def __init__(self, layer_sizes: list[int], num_wires: int | None = None) -> None:
            super().__init__()
            self.layer_sizes = list(layer_sizes)
            self.num_wires = num_wires or self.layer_sizes[-1]
            self.q_device = tq.QuantumDevice(n_wires=self.num_wires)
            self._build_ansatz()

        def _build_ansatz(self):
            self.func_list = []
            for idx, size in enumerate(self.layer_sizes):
                for wire in range(size):
                    self.func_list.append({"input_idx": [idx], "func": "ry", "wires": [wire]})

        @tq.static_support
        def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
            q_device.reset_states(x.shape[0])
            for info in self.func_list:
                params = x[:, info["input_idx"]] if op_name_dict[info["func"]].num_params else None
                func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
            for info in reversed(self.func_list):
                params = -y[:, info["input_idx"]] if op_name_dict[info["func"]].num_params else None
                func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

    class _QuantumKernel(tq.QuantumModule):
        def __init__(self, layer_sizes: list[int], num_wires: int | None = None) -> None:
            super().__init__()
            self.feature_map = HybridKernelGraphQNN._QuantumFeatureMap(layer_sizes, num_wires)
            self.q_device = self.feature_map.q_device

        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            x = x.reshape(1, -1)
            y = y.reshape(1, -1)
            self.feature_map(self.q_device, x, y)
            return torch.abs(self.q_device.states.view(-1)[0])

    # -------------------- Graph utilities --------------------
    @staticmethod
    def _random_linear(in_features: int, out_features: int) -> torch.Tensor:
        return torch.randn(out_features, in_features, dtype=torch.float32)

    @staticmethod
    def _random_training_data(weight: torch.Tensor, samples: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
        dataset: list[tuple[torch.Tensor, torch.Tensor]] = []
        for _ in range(samples):
            features = torch.randn(weight.size(1), dtype=torch.float32)
            target = weight @ features
            dataset.append((features, target))
        return dataset

    @staticmethod
    def _random_network(qnn_arch: Sequence[int], samples: int):
        weights: list[torch.Tensor] = []
        for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
            weights.append(HybridKernelGraphQNN._random_linear(in_f, out_f))
        target = weights[-1]
        training_data = HybridKernelGraphQNN._random_training_data(target, samples)
        return list(qnn_arch), weights, training_data, target

    @staticmethod
    def _feedforward(
        qnn_arch: Sequence[int],
        weights: Sequence[torch.Tensor],
        samples: Iterable[tuple[torch.Tensor, torch.Tensor]],
    ) -> list[list[torch.Tensor]]:
        stored: list[list[torch.Tensor]] = []
        for features, _ in samples:
            activations = [features]
            current = features
            for weight in weights:
                current = torch.tanh(weight @ current)
                activations.append(current)
            stored.append(activations)
        return stored

    @staticmethod
    def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float(torch.dot(a_norm, b_norm).item() ** 2)

    @staticmethod
    def _fidelity_adjacency(
        states: Sequence[torch.Tensor],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = HybridKernelGraphQNN.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    # -------------------- Public API --------------------
    def __init__(self, gamma: float = 1.0, q_layer_sizes: Sequence[int] | None = None, num_wires: int | None = None):
        self.gamma = gamma
        self.classical_kernel = self._ClassicalRBFKernel(gamma)
        self.quantum_kernel = self._QuantumKernel(q_layer_sizes or [1], num_wires)

    def classical_kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        return np.array([[self.classical_kernel(x, y).item() for y in b] for x in a])

    def quantum_kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        return np.array([[self.quantum_kernel(x, y).item() for y in b] for x in a])

    def feedforward(self, qnn_arch, weights, samples):
        return self._feedforward(qnn_arch, weights, samples)

    def fidelity_adjacency(self, states, threshold, secondary=None, secondary_weight=0.5):
        return self._fidelity_adjacency(states, threshold, secondary=secondary, secondary_weight=secondary_weight)

    def random_network(self, qnn_arch, samples):
        return self._random_network(qnn_arch, samples)

    def random_training_data(self, weight, samples):
        return self._random_training_data(weight, samples)

__all__ = [
    "HybridKernelGraphQNN",
    "_ClassicalRBFKernel",
    "_QuantumKernel",
    "_QuantumFeatureMap",
    "random_network",
    "random_training_data",
    "_feedforward",
    "state_fidelity",
    "_fidelity_adjacency",
]
