import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import itertools
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple, List

@dataclass
class LayerParams:
    """Parameters that describe a single photonic‑style layer."""
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]

def _clip(v: float, bound: float) -> float:
    return max(-bound, min(bound, v))

def _linear_layer(params: LayerParams, clip: bool) -> nn.Module:
    weight = torch.tensor([[params.bs_theta, params.bs_phi],
                           [params.squeeze_r[0], params.squeeze_r[1]]], dtype=torch.float32)
    bias = torch.tensor(params.phases, dtype=torch.float32)
    if clip:
        weight = weight.clamp(-5.0, 5.0)
        bias = bias.clamp(-5.0, 5.0)
    lin = nn.Linear(2, 2)
    with torch.no_grad():
        lin.weight.copy_(weight)
        lin.bias.copy_(bias)
    act = nn.Tanh()
    scale = torch.tensor(params.displacement_r, dtype=torch.float32)
    shift = torch.tensor(params.displacement_phi, dtype=torch.float32)
    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = lin
            self.activation = act
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.activation(self.linear(x))
            out = out * self.scale + self.shift
            return out
    return Layer()

def build_fraud_detection_program(input_params: LayerParams,
                                 layers: Iterable[LayerParams]) -> nn.Sequential:
    """Construct the classical linear stack that mirrors the photonic circuit."""
    modules: List[nn.Module] = [_linear_layer(input_params, clip=False)]
    modules.extend(_linear_layer(l, clip=True) for l in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

# ------------------------------------------------------------------
# Graph utilities (adapted from GraphQNN)
# ------------------------------------------------------------------
def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)

def fidelity_adjacency(states: Sequence[torch.Tensor],
                       threshold: float,
                       *,
                       secondary: float | None = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(range(len(states)))
    for (i, si), (j, sj) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(si, sj)
        if fid >= threshold:
            g.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            g.add_edge(i, j, weight=secondary_weight)
    return g

# ------------------------------------------------------------------
# Classical fraud detection model
# ------------------------------------------------------------------
class FraudDetectionHybrid(nn.Module):
    """Classical fraud‑detection model that chains a linear stack,
    a graph‑based aggregation, and a stacked LSTM."""
    def __init__(self,
                 input_dim: int = 2,
                 hidden_dim: int = 16,
                 lstm_layers: int = 2):
        super().__init__()
        # dummy params for the linear stack – in practice these would be learned
        dummy_params = LayerParams(0.0, 0.0, (0.0, 0.0),
                                   (0.0, 0.0), (0.0, 0.0),
                                   (0.0, 0.0), (0.0, 0.0))
        self.linear_stack = build_fraud_detection_program(dummy_params, [])
        self.lstm = nn.LSTM(input_dim, hidden_dim, lstm_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, 2)
        lin_out = self.linear_stack(x)
        lstm_out, _ = self.lstm(lin_out)
        out = self.fc(lstm_out[:, -1, :])
        return torch.sigmoid(out)

__all__ = ["LayerParams",
           "build_fraud_detection_program",
           "fidelity_adjacency",
           "FraudDetectionHybrid"]
