import torch
from torch import nn
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple, List

# --------------------------------------------------------------------------- #
# Classical QCNN inspired network
# --------------------------------------------------------------------------- #
class QCNNModel(nn.Module):
    """Stack of fully connected layers emulating the quantum convolution steps."""
    def __init__(self) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        # Output two logits so that the fraud‑detection block can consume them
        self.head = nn.Linear(4, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

def QCNN() -> QCNNModel:
    """Factory returning the configured :class:`QCNNModel`."""
    return QCNNModel()

# --------------------------------------------------------------------------- #
# Fraud‑detection style layers
# --------------------------------------------------------------------------- #
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

def _layer_from_params(params: FraudLayerParameters, clip: bool) -> nn.Module:
    weight = torch.tensor(
        [
            [params.bs_theta, params.bs_phi],
            [params.squeeze_r[0], params.squeeze_r[1]],
        ],
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

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            outputs = self.activation(self.linear(inputs))
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Create a sequential PyTorch model mirroring the layered structure."""
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

# --------------------------------------------------------------------------- #
# Unified classifier construction
# --------------------------------------------------------------------------- #
def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], List[int], List[int]]:
    """
    Construct a hybrid classical classifier that:
    1. Maps raw features to an 8‑dimensional feature space.
    2. Passes them through a QCNN‑style network.
    3. Applies a fraud‑detection post‑processing block.
    4. Produces binary logits.
    The signature mirrors the original seed for drop‑in compatibility.
    """
    # 1. Feature mapping
    preprocess = nn.Sequential(nn.Linear(num_features, 8), nn.ReLU())

    # 2. QCNN block
    qcnn = QCNN()

    # 3. Fraud‑detection layers (dummy parameters – replace with real ones for training)
    dummy_input = FraudLayerParameters(
        bs_theta=0.0,
        bs_phi=0.0,
        phases=(0.0, 0.0),
        squeeze_r=(0.0, 0.0),
        squeeze_phi=(0.0, 0.0),
        displacement_r=(0.0, 0.0),
        displacement_phi=(0.0, 0.0),
        kerr=(0.0, 0.0),
    )
    fraud_layers = [dummy_input for _ in range(depth)]
    fraud = build_fraud_detection_program(dummy_input, fraud_layers)

    # 4. Final classification head
    head = nn.Linear(1, 2)

    # Assemble full network
    network = nn.Sequential(
        preprocess,
        qcnn,
        fraud,
        head
    )

    # Metadata for the interface
    encoding = list(range(num_features))
    weight_sizes = []
    for module in network.modules():
        if isinstance(module, nn.Linear):
            weight_sizes.append(module.weight.numel() + module.bias.numel())
    observables = list(range(2))  # binary classification

    return network, encoding, weight_sizes, observables

__all__ = [
    "QCNN",
    "QCNNModel",
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "build_classifier_circuit",
]
