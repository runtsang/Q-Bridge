from __future__ import annotations

import numpy as np
import torch
from torch import nn
import networkx as nx
from dataclasses import dataclass
from typing import List, Tuple, Iterable

__all__ = ["ConvFusion", "ConvFusionParameters"]

@dataclass
class ConvFusionParameters:
    kernel_size: int = 2
    conv_threshold: float = 0.0
    fraud_params: List[Tuple[float, float, Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]]] | None = None
    graph_threshold: float = 0.95

class ConvFusion:
    """
    Classical hybrid module that applies a 2‑D convolution followed by a
    fraud‑detection style fully‑connected network.  It also offers graph
    construction utilities based on state fidelities.
    """

    def __init__(self, params: ConvFusionParameters | None = None):
        self.params = params or ConvFusionParameters()
        self._conv = nn.Conv2d(1, 1, kernel_size=self.params.kernel_size, bias=True)
        self._fraud = self._build_fraud_model()

    def _build_fraud_model(self) -> nn.Sequential:
        # Default fraud‑layer parameters (identity‑like)
        default = (1.0, 1.0, (0.0, 0.0), (0.0, 0.0), (0.0, 0.0),
                   (0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
        layers = self.params.fraud_params or [default]
        modules: List[nn.Module] = []

        for p in layers:
            weight = torch.tensor([[p[0], p[1]], [p[3][0], p[3][1]]], dtype=torch.float32)
            bias = torch.tensor(p[2], dtype=torch.float32)
            linear = nn.Linear(2, 2)
            with torch.no_grad():
                linear.weight.copy_(weight)
                linear.bias.copy_(bias)
            activation = nn.Tanh()
            scale = torch.tensor(p[4], dtype=torch.float32)
            shift = torch.tensor(p[5], dtype=torch.float32)

            class Layer(nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.linear = linear
                    self.activation = activation
                    self.register_buffer("scale", scale)
                    self.register_buffer("shift", shift)

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    out = self.activation(self.linear(x))
                    out = out * self.scale + self.shift
                    return out

            modules.append(Layer())

        modules.append(nn.Linear(2, 1))
        return nn.Sequential(*modules)

    def _state_fidelity(self, a: torch.Tensor, b: torch.Tensor) -> float:
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float(torch.dot(a_norm, b_norm).item() ** 2)

    def _fidelity_adjacency(self, states: List[torch.Tensor], threshold: float) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for i, a in enumerate(states):
            for j, b in enumerate(states):
                if j <= i:
                    continue
                fid = self._state_fidelity(a, b)
                if fid >= threshold:
                    graph.add_edge(i, j, weight=1.0)
        return graph

    def run(self, data: np.ndarray) -> float:
        """
        Apply the classical convolution, then the fraud‑detection network.
        Returns a scalar summarising the combined activation.
        """
        tensor = torch.as_tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        conv_out = self._conv(tensor)
        act = torch.sigmoid(conv_out - self.params.conv_threshold)
        out = act.mean()

        # Flatten conv output for the fraud model (2‑dim assumption)
        flat = conv_out.view(-1, 2)
        fraud_out = self._fraud(flat)
        return float((out + fraud_out.mean()).item())

    def graph_regularisation(self, data: np.ndarray) -> nx.Graph:
        """
        Build a fidelity graph from the hidden states produced by the
        fraud‑detection network.
        """
        tensor = torch.as_tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        conv_out = self._conv(tensor).detach()
        flat = conv_out.view(-1, 2)
        states: List[torch.Tensor] = []
        x = flat
        for module in self._fraud:
            x = module(x)
            states.append(x.detach())
        return self._fidelity_adjacency(states, self.params.graph_threshold)
