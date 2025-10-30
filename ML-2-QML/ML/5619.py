"""
Classical implementation of a depth‑controlled feed‑forward classifier
with a hybrid estimator that can also evaluate quantum circuits.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List, Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# --------------------------------------------------------------------------- #
#  Classical backbone – depth‑controlled feed‑forward network
# --------------------------------------------------------------------------- #
def build_classifier_circuit(
    num_features: int,
    depth: int,
    *,
    dropout_rate: float = 0.0,
    use_bias: bool = True,
) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """
    Construct a fully‑connected network that mirrors the quantum ansatz.

    Returns
    -------
    network : nn.Sequential
        The classifier network.
    encoding : list[int]
        List of indices that represent the input encoding.
    weight_sizes : list[int]
        Flattened number of parameters for each linear layer.
    observables : list[int]
        Dummy observable indices used by the estimator.
    """
    layers: List[nn.Module] = []
    in_dim = num_features

    # Optional dropout at the very beginning
    if dropout_rate > 0.0:
        layers.append(nn.Dropout(p=dropout_rate))

    # Fixed encoding – each feature is used once
    encoding = list(range(num_features))

    weight_sizes: List[int] = []

    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features, bias=use_bias)
        layers.append(linear)
        layers.append(nn.ReLU())
        weight_sizes.append(linear.weight.numel() + (linear.bias.numel() if use_bias else 0))
        in_dim = num_features

    # Final head that produces two logits
    head = nn.Linear(in_dim, 2, bias=use_bias)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + (head.bias.numel() if use_bias else 0))

    network = nn.Sequential(*layers)
    observables = list(range(2))

    return network, encoding, weight_sizes, observables

# --------------------------------------------------------------------------- #
#  Hybrid estimator – supports both torch and quantum backends
# --------------------------------------------------------------------------- #
class HybridEstimator:
    """
    Lightweight estimator that can evaluate either a PyTorch model or a
    quantum circuit that implements a ``run`` method returning expectation
    values for a list of parameter sets.
    """
    def __init__(self, model: nn.Module | "QuantumCircuitWrapper"):
        self.model = model

    def evaluate(
        self,
        observables: Iterable[Callable[[Tensor], Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """
        Forward all parameter sets through the model and compute the
        corresponding observables.

        * For a torch model this is a deterministic pass.
        * For a quantum circuit the ``QuantumCircuitWrapper`` is expected
          to expose a ``run`` method that returns expectation values.
        """
        if isinstance(self.model, nn.Module):
            return self._evaluate_torch(self.model, observables, parameter_sets)

        # Assume quantum circuit has a ``run`` method
        return self._evaluate_quantum(self.model, observables, parameter_sets)

    def _evaluate_torch(
        self,
        model: nn.Module,
        observables: Iterable[Callable[[Tensor], Tensor | float]],
        params: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        results: List[List[float]] = []
        model.eval()
        with torch.no_grad():
            for param_set in params:
                inputs = torch.as_tensor(param_set, dtype=torch.float32).unsqueeze(0)
                outputs = model(inputs)
                row: List[float] = []
                for obs in observables:
                    value = obs(outputs)
                    if isinstance(value, torch.Tensor):
                        value = float(value.mean().cpu())
                    else:
                        value = float(value)
                    row.append(value)
                results.append(row)
        return results

    def _evaluate_quantum(
        self,
        circuit: "QuantumCircuitWrapper",
        observables: Iterable[Callable[[Tensor], Tensor | float]],
        params: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """
        Calls the quantum circuit’s ``run`` method and interprets the
        returned values as expectations.  The observables are ignored
        because the circuit already encodes them.
        """
        raw = circuit.run(params)
        # Convert complex numbers to real if necessary
        results: List[List[float]] = [[float(val.real) if hasattr(val, "real") else float(val) for val in row] for row in raw]
        return results

# --------------------------------------------------------------------------- #
#  Unified classifier – exposes a single class in both modules
# --------------------------------------------------------------------------- #
class UnifiedClassifier(nn.Module):
    """
    A classifier that can be used in a purely classical setting or as a
    hybrid quantum‑classical model.  The class accepts a ``mode`` flag
    that determines whether the final head is a classical linear layer
    or a quantum circuit wrapper.
    """
    def __init__(
        self,
        num_features: int,
        depth: int,
        *,
        mode: str = "classical",
        quantum_circuit: "QuantumCircuitWrapper | None" = None,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.mode = mode
        self.backbone, self.encoding, self.weight_sizes, self.observables = build_classifier_circuit(
            num_features, depth, dropout_rate=dropout_rate
        )
        if mode == "quantum" and quantum_circuit is not None:
            self.head = quantum_circuit
        else:
            # Classical head that outputs two logits
            self.head = nn.Linear(self.backbone[-1].out_features, 2)

        # The estimator will delegate to the appropriate backend
        self.estimator = HybridEstimator(self.head if mode == "quantum" else self.backbone)

    def forward(self, x: Tensor) -> Tensor:
        # Pass through the backbone
        features = self.backbone(x)
        if self.mode == "quantum":
            # The quantum head expects a list of parameters per sample
            # Here we simply feed the feature vector to the circuit run method
            return self.head.run(features.squeeze().tolist())
        else:
            logits = self.head(features)
            probs = F.softmax(logits, dim=-1)
            return probs

    def evaluate(
        self,
        observables: Iterable[Callable[[Tensor], Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """
        Delegate evaluation to the underlying estimator.  The estimator
        automatically handles classical or quantum backends.
        """
        return self.estimator.evaluate(observables, parameter_sets)

__all__ = ["build_classifier_circuit", "HybridEstimator", "UnifiedClassifier"]
