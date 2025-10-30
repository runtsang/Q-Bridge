"""Hybrid fraud detection model combining a quantum feature extractor and classical layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Dict, Any

import torch
from torch import nn


@dataclass
class FraudLayerParameters:
    """Encapsulates the parameters for one photonic‑style classical layer."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


def _clip(value: float, bound: float) -> float:
    """Clamp a scalar to a symmetric interval."""
    return max(-bound, min(bound, value))


def _build_classical_layer(params: FraudLayerParameters, clip: bool) -> nn.Module:
    """Construct a single classical layer from the provided parameters."""
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

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            out = self.activation(self.linear(inputs))
            out = out * self.scale + self.shift
            return out

    return Layer()


class FraudDetectionHybrid(nn.Module):
    """Hybrid fraud detection model.

    Parameters
    ----------
    quantum_circuit : Any
        Quantum circuit object (e.g., Qiskit QuantumCircuit).
    quantum_observable : Any
        Observable used by the quantum estimator.
    quantum_input_params : Sequence[Parameter]
        Parameters that encode the input features.
    quantum_weight_params : Sequence[Parameter]
        Parameters that encode the trainable weights.
    quantum_estimator : Any
        Estimator object capable of running the circuit (e.g., StatevectorEstimator).
    input_params : FraudLayerParameters
        Parameters for the first classical layer.
    layers : Iterable[FraudLayerParameters]
        Parameters for subsequent classical layers.
    """
    def __init__(
        self,
        quantum_circuit: Any,
        quantum_observable: Any,
        quantum_input_params: Sequence[Any],
        quantum_weight_params: Sequence[Any],
        quantum_estimator: Any,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
    ) -> None:
        super().__init__()
        self.quantum_circuit = quantum_circuit
        self.quantum_observable = quantum_observable
        self.quantum_input_params = quantum_input_params
        self.quantum_weight_params = quantum_weight_params
        self.quantum_estimator = quantum_estimator

        # Build the classical stack
        self.classical_stack = nn.Sequential(
            _build_classical_layer(input_params, clip=False),
            *(_build_classical_layer(l, clip=True) for l in layers),
            nn.Linear(2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that first evaluates the quantum circuit to obtain
        expectation values, then feeds them through the classical stack.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 2).  The first column is mapped to
            quantum input parameters, the second to quantum weight parameters.

        Returns
        -------
        torch.Tensor
            Output predictions of shape (batch, 1).
        """
        # Prepare parameter dictionary for the quantum estimator
        param_dict: Dict[Any, float] = {
            p: _clip(v.item(), 5.0)
            for p, v in zip(self.quantum_input_params, x[:, 0])
        }
        param_dict.update({
            p: _clip(v.item(), 5.0)
            for p, v in zip(self.quantum_weight_params, x[:, 1])
        })

        # Run the quantum estimator (expectation values)
        expvals = self.quantum_estimator.run(
            self.quantum_circuit,
            param_dict,
            observables=self.quantum_observable,
        )
        # Convert to torch tensor
        q_features = torch.tensor(expvals, dtype=torch.float32, device=x.device)

        # Classical post‑processing
        return self.classical_stack(q_features)


__all__ = ["FraudLayerParameters", "FraudDetectionHybrid"]
