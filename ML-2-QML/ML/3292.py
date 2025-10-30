import itertools
import numpy as np
import torch
import networkx as nx
from typing import Iterable, List, Sequence, Tuple

Tensor = torch.Tensor

class GraphQNN:
    """Classical graph neural network mirroring its quantum counterpart.

    Features:
        * Random weight initialization per layer.
        * Feed‑forward propagation returning all intermediate activations.
        * Fidelity‑based adjacency graph construction.
        * Drop‑in replacement for a fully‑connected quantum layer (FCL).
    """

    def __init__(self, arch: Sequence[int]):
        self.arch = list(arch)

    # ------------------- Random network utilities --------------------
    def _random_linear(self, in_f: int, out_f: int) -> Tensor:
        return torch.randn(out_f, in_f, dtype=torch.float32)

    def random_network(self, samples: int) -> Tuple[List[int], List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
        weights = [self._random_linear(i, o) for i, o in zip(self.arch[:-1], self.arch[1:])]
        target = weights[-1]
        train = [(torch.randn(target.size(1)), target @ torch.randn(target.size(1))) for _ in range(samples)]
        return self.arch, weights, train, target

    # ------------------- Forward propagation -------------------------
    def feedforward(self, weights: Iterable[Tensor], samples: Iterable[Tuple[Tensor, Tensor]]) -> List[List[Tensor]]:
        outputs = []
        for feat, _ in samples:
            acts = [feat]
            cur = feat
            for w in weights:
                cur = torch.tanh(w @ cur)
                acts.append(cur)
            outputs.append(acts)
        return outputs

    # ------------------- Fidelity helpers ----------------------------
    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        a_n = a / (torch.norm(a) + 1e-12)
        b_n = b / (torch.norm(b) + 1e-12)
        return float((a_n @ b_n).item() ** 2)

    @staticmethod
    def fidelity_adjacency(states: Sequence[Tensor], threshold: float,
                           *, secondary: float | None = None,
                           secondary_weight: float = 0.5) -> nx.Graph:
        G = nx.Graph()
        G.add_nodes_from(range(len(states)))
        for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNN.state_fidelity(s_i, s_j)
            if fid >= threshold:
                G.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                G.add_edge(i, j, weight=secondary_weight)
        return G

    # ------------------- Classical FCL -------------------------------
    def FCL(self, n_features: int = 1):
        """Return a tiny torch module mimicking the quantum FCL example."""
        class _FCL(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(n_features, 1)

            def run(self, thetas: Iterable[float]) -> np.ndarray:
                x = torch.tensor(list(thetas), dtype=torch.float32).view(-1, 1)
                return torch.tanh(self.linear(x)).mean().detach().numpy()

        return _FCL()

__all__ = [
    "GraphQNN",
]
