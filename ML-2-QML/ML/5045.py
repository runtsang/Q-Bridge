"""UnifiedClassifier – classical PyTorch implementation.

The module exposes a flexible head interface that can be switched at construction time
between a dense linear head, a sampler‑style head, a fraud‑detection head or a
variational‑classifier head.  It also exposes a small API to export the metadata
required by the corresponding quantum implementation so that the two modules
are guaranteed to remain in sync.

Typical usage:
    clf = UnifiedClassifier(head_type="sampler")
    preds = clf(torch.randn(8, 3, 32, 32))
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Tuple, List, Dict, Any

# --------------------------------------------------------------------------- #
# 1.  Heads – fully classical implementations
# --------------------------------------------------------------------------- #

class HybridFunction(nn.Module):
    """Simple sigmoid head used by the dense classifier."""
    def __init__(self, shift: float = 0.0) -> None:
        super().__init__()
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x + self.shift)


class DenseHead(nn.Module):
    """Dense fully‑connected head with a single linear layer followed by a sigmoid."""
    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.fn = HybridFunction(shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(self.linear(x))


class SamplerHead(nn.Module):
    """Replicates the structure of the classical sampler network from Pair 2."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(x), dim=-1)


class FraudHead(nn.Module):
    """Creates a sequential network mirroring the fraud‑detection circuit."""
    def __init__(self, params: Iterable[Dict[str, Any]]) -> None:
        super().__init__()
        self.net = build_fraud_detection_program_from_params(params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ClassifierHead(nn.Module):
    """Feed‑forward classifier built from the classical circuit factory (Pair 4)."""
    def __init__(self, num_features: int, depth: int) -> None:
        super().__init__()
        self.net, *_ = build_classifier_circuit(num_features, depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# --------------------------------------------------------------------------- #
# 2.  Helper functions for fraud‑detection head
# --------------------------------------------------------------------------- #

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
        dtype=torch.float32,
    )
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

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            outputs = self.activation(self.linear(inputs))
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()


def build_fraud_detection_program_from_params(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


# --------------------------------------------------------------------------- #
# 3.  Helper functions for the classical classifier factory (Pair 4)
# --------------------------------------------------------------------------- #

def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, List[int], List[int], List[int]]:
    """Return a classical feed‑forward network and its metadata."""
    layers: List[nn.Module] = []
    in_dim = num_features
    weight_sizes: List[int] = []

    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.extend([linear, nn.ReLU()])
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features

    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)
    encoding = list(range(num_features))
    observables = list(range(2))
    return network, encoding, weight_sizes, observables


# --------------------------------------------------------------------------- #
# 4.  UnifiedClassifier – the public API
# --------------------------------------------------------------------------- #

class UnifiedClassifier(nn.Module):
    """Flexible CNN‑based classifier with multiple interchangeable heads.

    Parameters
    ----------
    head_type : str, optional
        One of ``'dense'``, ``'sampler'``, ``'fraud'`` or ``'classifier'``.
    head_kwargs : dict, optional
        Additional keyword arguments forwarded to the chosen head constructor.
    """
    def __init__(self, head_type: str = "dense", **head_kwargs) -> None:
        super().__init__()
        self.head_type = head_type
        self.base = self._build_base()
        self.head = self._build_head(head_type, **head_kwargs)

    # --------------------- base CNN ---------------------------------------

    def _build_base(self) -> nn.Module:
        conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        pool = nn.MaxPool2d(kernel_size=2, stride=1)
        drop1 = nn.Dropout2d(p=0.2)
        drop2 = nn.Dropout2d(p=0.5)
        fc1 = nn.Linear(55815, 120)
        fc2 = nn.Linear(120, 84)
        fc3 = nn.Linear(84, 1)

        return nn.Sequential(
            conv1, nn.ReLU(), pool, drop1,
            conv2, nn.ReLU(), pool, drop1,
            nn.Flatten(), fc1, nn.ReLU(), drop2,
            fc2, nn.ReLU(), fc3
        )

    # --------------------- head construction --------------------------------

    def _build_head(self, head_type: str, **kwargs) -> nn.Module:
        if head_type == "dense":
            return DenseHead(in_features=1, **kwargs)
        if head_type == "sampler":
            return SamplerHead()
        if head_type == "fraud":
            params = kwargs.get("params", [])
            return FraudHead(params)
        if head_type == "classifier":
            num_features = kwargs.get("num_features", 10)
            depth = kwargs.get("depth", 2)
            return ClassifierHead(num_features, depth)
        raise ValueError(f"Unsupported head_type: {head_type}")

    # --------------------- forward -----------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.base(x)
        logits = self.head(features)
        probs = F.softmax(logits, dim=-1)
        return probs

    # --------------------- utility -----------------------------------------

    def export_quantum_metadata(self) -> Dict[str, Any] | None:
        """Return a dictionary describing the quantum circuit needed for the head.
        This is used by the quantum module to reconstruct the same computation.
        """
        if self.head_type in ("sampler", "fraud", "classifier"):
            # Each head can expose a simple descriptor; for the purposes of this
            # example we just return the head type and the parameters used.
            return {"head_type": self.head_type, "params": getattr(self.head, "net", None)}
        return None


__all__ = ["UnifiedClassifier", "FraudLayerParameters", "build_classifier_circuit"]
