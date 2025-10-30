"""HybridClassifier – classical implementation.

This module mirrors the quantum helper interface while providing a fully
trainable PyTorch network.  The network architecture, metadata, and
evaluation utilities are directly inspired by the classical seeds and the
FastEstimator pattern, but are extended to support fraud‑detection style
parameterisation.

The public API is intentionally minimal – only the constructor and a
``evaluate`` method – so that it can be dropped into any training pipeline
without side effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import torch
from torch import nn

# --------------------------------------------------------------------------- #
#   Data container – fraud‑detection style parameters
# --------------------------------------------------------------------------- #
@dataclass
class FraudLayerParameters:
    """Parameters for a single fully‑connected layer used by the hybrid model."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


# --------------------------------------------------------------------------- #
#   Utility – build the classical classifier network
# --------------------------------------------------------------------------- #
def _build_classical_circuit(num_features: int, depth: int) -> tuple[nn.Module, list[int], list[int], list[int]]:
    """Return a sequential net and associated metadata."""
    layers: List[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: List[int] = []

    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.extend([linear, nn.ReLU()])
        weight_sizes.append(int(linear.weight.numel() + linear.bias.numel()))
        in_dim = num_features

    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(int(head.weight.numel() + head.bias.numel()))

    network = nn.Sequential(*layers)
    observables = list(range(2))
    return network, encoding, weight_sizes, observables


# --------------------------------------------------------------------------- #
#   Fast estimator – evaluates batches of parameters and observables
# --------------------------------------------------------------------------- #
def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Guarantee a 2‑D batch tensor for a single point."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastEstimator:
    """Deterministic evaluator with optional Gaussian shot noise."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []

        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.model(inputs)
                row: List[float] = []

                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        scalar = float(val.mean().cpu())
                    else:
                        scalar = float(val)
                    row.append(scalar)

                results.append(row)

        if shots is None:
            return results

        rng = torch.Generator().manual_seed(seed) if seed is not None else None
        noisy: List[List[float]] = []
        for row in results:
            noisy_row = [float(torch.randn(1, generator=rng).item() / shots + mean) for mean in row]
            noisy.append(noisy_row)

        return noisy


# --------------------------------------------------------------------------- #
#   Main hybrid classifier – classical implementation
# --------------------------------------------------------------------------- #
class HybridClassifier:
    """
    Classical hybrid classifier.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input feature vector.
    depth : int
        Number of hidden layers in the feed‑forward net.
    fraud_params : Iterable[FraudLayerParameters] | None
        Optional fraud‑detection style parameters used to initialise the
        first layer.  When supplied, the first layer uses the exact weights
        and biases derived from the parameters; subsequent layers are
        standard learnable layers.
    """

    def __init__(
        self,
        num_features: int,
        depth: int,
        *,
        fraud_params: Iterable[FraudLayerParameters] | None = None,
    ) -> None:
        # Build the base network
        base_net, encoding, weight_sizes, observables = _build_classical_circuit(num_features, depth)
        self._encoding = encoding
        self._weight_sizes = weight_sizes
        self._observables = observables

        # If fraud‑detection parameters are supplied, overwrite the first layer
        if fraud_params is not None:
            first_layer = _layer_from_params(next(iter(fraud_params)), clip=False)
            base_net[0] = first_layer

        self.model = base_net
        self.estimator = FastEstimator(self.model)

    @property
    def encoding(self) -> list[int]:
        """Indices of input features used for encoding."""
        return self._encoding

    @property
    def weight_sizes(self) -> list[int]:
        """Number of trainable parameters per layer."""
        return self._weight_sizes

    @property
    def observables(self) -> list[int]:
        """Indices of output logits."""
        return self._observables

    def evaluate(
        self,
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """
        Evaluate the classifier on a batch of parameters.

        Parameters
        ----------
        parameter_sets : Sequence[Sequence[float]]
            Each inner sequence is a flattened vector of model parameters.
        shots : int | None, optional
            If provided, inject Gaussian noise with variance 1/shots.
        seed : int | None, optional
            Seed for reproducible noise.

        Returns
        -------
        List[List[float]]
            List of prediction scores for each parameter set.
        """
        return self.estimator.evaluate(
            observables=[lambda out: out[:, 0], lambda out: out[:, 1]],
            parameter_sets=parameter_sets,
            shots=shots,
            seed=seed,
        )

# --------------------------------------------------------------------------- #
#   Helper – build a fraud‑layer from parameters
# --------------------------------------------------------------------------- #
def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
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
