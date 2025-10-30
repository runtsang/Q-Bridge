import itertools
import numpy as np
import torch
import networkx as nx
from torch import nn
from typing import Sequence, List, Tuple, Iterable

class GraphQNNHybrid:
    """Hybrid Graph QNN class with classical backend.

    Combines the neural‑network feed‑forward logic from the original GraphQNN
    seed with auxiliary utilities (conv filter, fraud‑detection model)
    that mirror the quantum counterparts.  All operations are purely
    NumPy/Torch and therefore compatible with any CPU or GPU backend.
    """

    def __init__(self, arch: Sequence[int]) -> None:
        self.arch = list(arch)

    def random_network(self, samples: int) -> Tuple[List[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        weights = []
        for in_f, out_f in zip(self.arch[:-1], self.arch[1:]):
            weights.append(torch.randn(out_f, in_f, dtype=torch.float32))
        target = weights[-1]
        training = []
        for _ in range(samples):
            x = torch.randn(target.shape[1], dtype=torch.float32)
            y = target @ x
            training.append((x, y))
        return weights, training, target

    def feedforward(self, weights: Sequence[torch.Tensor], samples: Iterable[Tuple[torch.Tensor, torch.Tensor]]) -> List[List[torch.Tensor]]:
        activations = []
        for x, _ in samples:
            layer_out = [x]
            current = x
            for w in weights:
                current = torch.tanh(w @ current)
                layer_out.append(current)
            activations.append(layer_out)
        return activations

    @staticmethod
    def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
        a_n = a / (torch.norm(a) + 1e-12)
        b_n = b / (torch.norm(b) + 1e-12)
        return float((a_n @ b_n).item() ** 2)

    @staticmethod
    def fidelity_adjacency(states: Sequence[torch.Tensor], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNNHybrid.state_fidelity(s_i, s_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    @staticmethod
    def Conv(kernel_size: int = 2, threshold: float = 0.0) -> nn.Module:
        """Return a classical convolutional filter mimicking the quantum quanvolution."""
        class ConvFilter(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

            def run(self, data: np.ndarray) -> float:
                tensor = torch.as_tensor(data, dtype=torch.float32).view(1, 1, self.conv.kernel_size, self.conv.kernel_size)
                logits = self.conv(tensor)
                activations = torch.sigmoid(logits - threshold)
                return activations.mean().item()
        return ConvFilter()

    @staticmethod
    def FraudDetection(input_params, layers):
        """Build a classical fraud‑detection sequential model from photonic parameters."""
        from torch import nn
        import torch
        from typing import Iterable, Sequence
        from dataclasses import dataclass

        @dataclass
        class FraudLayerParameters:
            bs_theta: float
            bs_phi: float
            phases: tuple[float, float]
            squeeze_r: tuple[float, float]
            squeeze_phi: tuple[float, float]
            displacement_r: tuple[float, float]
            displacement_phi: tuple[float, float]
            kerr: tuple[float, float]

        def _clip(value: float, bound: float) -> float:
            return max(-bound, min(bound, value))

        def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
            weight = torch.tensor([[params.bs_theta, params.bs_phi],
                                   [params.squeeze_r[0], params.squeeze_r[1]]], dtype=torch.float32)
            bias = torch.tensor(params.phases, dtype=torch.float32)
            if clip:
                weight = weight.clamp(-5.0, 5.0)
                bias = bias.clamp(-5.0, 5.0)
            linear = nn.Linear(2, 2)
            with torch.no_grad():
                linear.weight.copy_(weight)
                linear.bias.copy_(bias)
            activation = nn.Tanh()
            scale = torch.tensor(params.displacement_r, dtype=torch.float32)
            shift = torch.tensor(params.displacement_phi, dtype=torch.float32)

            class Layer(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = linear
                    self.activation = activation
                    self.register_buffer("scale", scale)
                    self.register_buffer("shift", shift)

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    out = self.activation(self.linear(x))
                    out = out * self.scale + self.shift
                    return out
            return Layer()

        modules = [_layer_from_params(input_params, clip=False)]
        modules.extend(_layer_from_params(l, clip=True) for l in layers)
        modules.append(nn.Linear(2, 1))
        return nn.Sequential(*modules)

__all__ = ["GraphQNNHybrid"]
