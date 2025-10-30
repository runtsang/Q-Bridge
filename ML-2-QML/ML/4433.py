import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from dataclasses import dataclass
from typing import Iterable, Tuple

@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    weight = torch.tensor(
        [[params.bs_theta, params.bs_phi],
         [params.squeeze_r[0], params.squeeze_r[1]]],
        dtype=torch.float32)
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
        def __init__(self) -> None:
            super().__init__()
            self.linear = linear
            self.activation = activation
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            outputs = self.activation(self.linear(inputs))
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)

def fidelity_adjacency(
    states: Iterable[torch.Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for i, state_i in enumerate(states):
        for j, state_j in enumerate(states):
            if j <= i:
                continue
            fid = state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
    return graph

class FraudDetectionHybrid(nn.Module):
    """
    Classical fraud‑detection backbone that fuses:
    * a photonic‑style feed‑forward network,
    * an LSTM for sequential transaction patterns,
    * a graph‑based similarity layer for embedding clustering.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        seq_len: int,
        n_layers: int = 1,
        graph_threshold: float = 0.8,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=n_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        self.graph_threshold = graph_threshold
        self.fraud_params = FraudLayerParameters(
            bs_theta=0.5,
            bs_phi=0.3,
            phases=(0.1, -0.1),
            squeeze_r=(0.2, 0.2),
            squeeze_phi=(0.0, 0.0),
            displacement_r=(0.5, 0.5),
            displacement_phi=(0.0, 0.0),
            kerr=(0.0, 0.0),
        )
        self.feedforward = build_fraud_detection_program(self.fraud_params, [])

    def build_graph(self, embeddings: torch.Tensor) -> nx.Graph:
        """
        Construct a similarity graph from transaction embeddings.
        """
        return fidelity_adjacency(embeddings, self.graph_threshold)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor
            Shape (batch, seq_len, input_dim)

        Returns
        -------
        Tensor
            Fraud probability of shape (batch, 1)
        """
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        ff_out = self.feedforward(last_hidden)
        combined = torch.cat([last_hidden, ff_out], dim=1)
        logits = self.fc(combined)
        return logits

__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "state_fidelity",
    "fidelity_adjacency",
    "FraudDetectionHybrid",
]
