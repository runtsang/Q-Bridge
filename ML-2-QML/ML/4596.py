"""FraudDetectionModel – a hybrid classical‑quantum fraud detector.

The module builds a classical neural network from the `FraudLayerParameters`
structure (a close analogue to the photonic circuit) and optionally
augments its output with a quantum‑kernel embedding.  A lightweight
graph‑convolutional layer (derived from the fidelity‑adjacency graph of
the intermediate representations) is inserted before the final linear
classifier.

The code is intentionally lightweight: training loops, data loaders and
regularisation utilities are left to the user, but the public API
(`FraudDetectionModel`) exposes a convenient `forward`, `fit` and
`predict` interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Optional

import torch
from torch import nn
import numpy as np

# --------------------------------------------------------------------------- #
# 1. Classical layer definitions – photonic‑style multilayer network
# --------------------------------------------------------------------------- #

@dataclass
class FraudLayerParameters:
    """Parameters of a single photonic‑style layer.

    The parameters mirror the ones used in the Strawberry Fields reference,
    but are interpreted in a purely classical neural‑network context.
    """
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]
    clip: bool = False  # whether to clip the weight matrix

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _layer_from_params(params: FraudLayerParameters) -> nn.Module:
    """Return a single linear‑Tanh‑scale layer."""
    weight = torch.tensor(
        [
            [params.bs_theta, params.bs_phi],
            [params.squeeze_r[0], params.squeeze_r[1]],
        ],
        dtype=torch.float32,
    )
    bias = torch.tensor(params.phases, dtype=torch.float32)

    if params.clip:
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
            out = self.activation(self.linear(inputs))
            return out * self.scale + self.shift

    return Layer()

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Create a torch.nn.Sequential mirroring the photonic circuit."""
    modules = [_layer_from_params(input_params)]
    modules.extend(_layer_from_params(layer) for layer in layers)
    modules.append(nn.Linear(2, 1))  # final output layer
    return nn.Sequential(*modules)

# --------------------------------------------------------------------------- #
# 2. Graph‑based feature aggregation – fidelity adjacency from GraphQNN
# --------------------------------------------------------------------------- #

def _fidelity_adjacency(states: Sequence[torch.Tensor], threshold: float) -> torch.Tensor:
    """Return a symmetric adjacency matrix derived from pairwise cosine
    similarities (treated as fidelities).  The function is a lightweight
    drop‑in replacement for the networkx implementation in the original
    GraphQNN reference.
    """
    n = len(states)
    mat = torch.zeros((n, n), dtype=torch.float32)
    for i in range(n):
        for j in range(i + 1, n):
            fid = torch.dot(states[i], states[j]) / (
                torch.norm(states[i]) * torch.norm(states[j]) + 1e-12
            )
            if fid.item() >= threshold:
                mat[i, j] = mat[j, i] = 1.0
    return mat

class GraphConvLayer(nn.Module):
    """A simple graph convolution that aggregates neighbour states
    weighted by the fidelity adjacency matrix.
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        aggregated = adjacency @ features
        return aggregated @ self.weight.t()

# --------------------------------------------------------------------------- #
# 3. Quantum kernel module – interface to the QML implementation
# --------------------------------------------------------------------------- #

try:
    from.QuantumKernelMethod import Kernel as QuantumKernel  # pragma: no cover
except Exception:  # pragma: no cover
    QuantumKernel = None  # fallback if the QML module is missing

# --------------------------------------------------------------------------- #
# 4. Main hybrid model
# --------------------------------------------------------------------------- #

class FraudDetectionModel(nn.Module):
    """Hybrid fraud detector combining classical layers, graph‑conv, and a
    quantum kernel.
    """

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layer_params: Sequence[FraudLayerParameters],
        *,
        use_quantum_kernel: bool = False,
        reference_vectors: Optional[torch.Tensor] = None,
        graph_threshold: float = 0.9,
        graph_out_features: int = 4,
    ) -> None:
        super().__init__()
        self.classical = build_fraud_detection_program(input_params, layer_params)
        self.use_qk = use_quantum_kernel and QuantumKernel is not None

        if self.use_qk:
            if reference_vectors is None:
                raise ValueError("reference_vectors must be provided when using a quantum kernel.")
            self.reference_vectors = reference_vectors  # shape (N, D)
            self.qkernel = QuantumKernel()
        else:
            self.reference_vectors = None
            self.qkernel = None

        # Graph convolution on the output of the classical net
        self.graph_conv = GraphConvLayer(1, graph_out_features)
        self.classifier = nn.Linear(graph_out_features + (reference_vectors.shape[1] if self.use_qk else 0), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass producing a scalar logit."""
        # Classical feature extraction
        cls_out = self.classical(x).view(-1, 1)  # shape (batch, 1)

        # Graph aggregation
        with torch.no_grad():
            # Compute fidelity adjacency over the batch
            adjacency = _fidelity_adjacency(cls_out.squeeze(-1).tolist(), threshold=0.8)
            adjacency = torch.tensor(adjacency, device=cls_out.device, dtype=cls_out.dtype)

        graph_feat = self.graph_conv(cls_out, adjacency)  # (batch, out_features)

        # Quantum kernel augmentation
        if self.use_qk:
            # Compute kernel vector between batch and reference set
            qk_vec = torch.stack([self.qkernel(xi, self.reference_vectors[i]) for i, xi in enumerate(x)])
            combined = torch.cat([graph_feat, qk_vec], dim=1)
        else:
            combined = graph_feat

        # Final classification
        return self.classifier(combined).squeeze(-1)

    def fit(self, x: torch.Tensor, y: torch.Tensor, lr: float = 1e-3, epochs: int = 200):
        """A minimal training loop using Adam and MSE loss."""
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        for _ in range(epochs):
            optimizer.zero_grad()
            preds = self.forward(x)
            loss = loss_fn(preds, y)
            loss.backward()
            optimizer.step()

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Binary prediction based on the trained logit."""
        logits = self.forward(x)
        return (logits > threshold).long()

__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "FraudDetectionModel",
]
