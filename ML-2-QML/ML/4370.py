import itertools
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Sequence, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass

# ----------------------------------------------------------------------
#  Classical utilities: random networks, feed‑forward, fidelity graph
# ----------------------------------------------------------------------
def _random_linear(in_features: int, out_features: int) -> torch.Tensor:
    """Return a random weight matrix with shape (out_features, in_features)."""
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: torch.Tensor, samples: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Generate synthetic training pairs (x, Wx) for the target weight."""
    dataset: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(samples):
        x = torch.randn(weight.size(1), dtype=torch.float32)
        y = weight @ x
        dataset.append((x, y))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    """Create a random classical MLP and its training data."""
    weights: List[torch.Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight

def feedforward(qnn_arch: Sequence[int],
                weights: Sequence[torch.Tensor],
                samples: Iterable[Tuple[torch.Tensor, torch.Tensor]]) -> List[List[torch.Tensor]]:
    """Forward pass through the classical MLP, storing activations per layer."""
    activations_per_sample: List[List[torch.Tensor]] = []
    for x, _ in samples:
        activations: List[torch.Tensor] = [x]
        current = x
        for w in weights:
            current = torch.tanh(w @ current)
            activations.append(current)
        activations_per_sample.append(activations)
    return activations_per_sample

def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Normalized squared dot product between two vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float((a_norm @ b_norm).item() ** 2)

def fidelity_adjacency(states: Sequence[torch.Tensor],
                       threshold: float,
                       *,
                       secondary: Optional[float] = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
    """Build a weighted graph where edges represent state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(s_i, s_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

# ----------------------------------------------------------------------
#  Hybrid classifier – a dense head that mimics a quantum expectation
# ----------------------------------------------------------------------
class HybridFunction(torch.autograd.Function):
    """Differentiable sigmoid activation that mimics a quantum expectation head."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float) -> torch.Tensor:  # type: ignore[override]
        outputs = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        (outputs,) = ctx.saved_tensors
        grad_inputs = grad_output * outputs * (1 - outputs)
        return grad_inputs, None

class Hybrid(nn.Module):
    """Dense head that replaces a quantum circuit."""
    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        logits = inputs.view(inputs.size(0), -1)
        return HybridFunction.apply(logits, self.shift)

class QCNet(nn.Module):
    """CNN followed by a hybrid dense head."""
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.hybrid = Hybrid(self.fc3.out_features, shift=0.0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = F.relu(self.conv1(inputs))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        probs = self.hybrid(x)
        return torch.cat((probs, 1 - probs), dim=-1)

# ----------------------------------------------------------------------
#  Fully‑connected quantum layer – classical surrogate
# ----------------------------------------------------------------------
class FCL(nn.Module):
    """Simple fully‑connected layer that mimics the quantum FCL."""
    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        values = torch.tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        expectation = torch.tanh(self.linear(values)).mean(dim=0)
        return expectation.detach().numpy()

# ----------------------------------------------------------------------
#  Fraud‑detection stack – classical photonic analogue
# ----------------------------------------------------------------------
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
        def __init__(self) -> None:
            super().__init__()
            self.linear = linear
            self.activation = activation
            self.register_buffer('scale', scale)
            self.register_buffer('shift', shift)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            outputs = self.activation(self.linear(inputs))
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()

def build_fraud_detection_program(input_params: FraudLayerParameters,
                                  layers: Iterable[FraudLayerParameters]) -> nn.Sequential:
    """Create a sequential PyTorch model mirroring the layered structure."""
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

# ----------------------------------------------------------------------
#  Unified interface
# ----------------------------------------------------------------------
class GraphQNNGen047:
    """Unified interface for classical graph‑based neural networks.

    The class bundles helper constructors for the classical MLP, the hybrid
    CNN‑quantum classifier, a fully‑connected quantum layer, and a fraud‑
    detection stack.  All components are available as static methods or
    attributes so that the same API can be used in both the ML and QML
    modules.
    """
    HybridFunction = HybridFunction
    Hybrid = Hybrid
    QCNet = QCNet
    FCL = FCL
    FraudLayerParameters = FraudLayerParameters
    build_fraud_detection_program = staticmethod(build_fraud_detection_program)

    @staticmethod
    def random_network(qnn_arch: Sequence[int], samples: int):
        return random_network(qnn_arch, samples)

    @staticmethod
    def feedforward(qnn_arch: Sequence[int], weights: Sequence[torch.Tensor],
                    samples: Iterable[Tuple[torch.Tensor, torch.Tensor]]):
        return feedforward(qnn_arch, weights, samples)

    @staticmethod
    def fidelity_adjacency(states: Sequence[torch.Tensor], threshold: float,
                           *, secondary: Optional[float] = None,
                           secondary_weight: float = 0.5):
        return fidelity_adjacency(states, threshold,
                                  secondary=secondary,
                                  secondary_weight=secondary_weight)

__all__ = [
    "GraphQNNGen047",
    "HybridFunction",
    "Hybrid",
    "QCNet",
    "FCL",
    "FraudLayerParameters",
    "build_fraud_detection_program",
]
