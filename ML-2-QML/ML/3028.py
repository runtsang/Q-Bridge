"""Hybrid estimator combining classical fraud‑detection layers with a Qiskit quantum circuit.

The class `HybridEstimator` first passes inputs through a stack of
classical `FraudLayerParameters`‑based layers (mirroring the photonic
implementation).  The resulting feature vector is collapsed to a single
scalar that is used as the input parameter of a tiny one‑qubit
parameter‑ized circuit.  The circuit weight is a learnable PyTorch
parameter that is mapped to a Qiskit `Parameter` object at every
forward pass.  The expectation value of the `Y` observable is then
treated as the quantum feature and fed into a final linear layer to
produce the regression output.

This design demonstrates how a classical neural network can serve as a
feature extractor for a quantum sub‑model while keeping the overall
model differentiable in PyTorch.  It also showcases a clean interface
between PyTorch and Qiskit that can be extended to larger circuits
or different observables.
"""

from __future__ import annotations

import torch
from torch import nn

from dataclasses import dataclass
from typing import Iterable, Sequence

# --------------------------------------------------------------------------- #
# Classical fraud‑detection layers (photonic analogue)
# --------------------------------------------------------------------------- #

@dataclass
class FraudLayerParameters:
    """Parameters describing a fully connected layer in the classical model."""
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
# Quantum sub‑model
# --------------------------------------------------------------------------- #

# Import quantum helper from the separate quantum module
from EstimatorQNN__gen301_qml import build_quantum_estimator

# --------------------------------------------------------------------------- #
# Hybrid model
# --------------------------------------------------------------------------- #

class HybridEstimator(nn.Module):
    """
    Combines a classical fraud‑detection feature extractor with a
    one‑qubit quantum estimator.  The quantum weight is a learnable
    PyTorch parameter that is injected into the Qiskit circuit at each
    forward pass.
    """

    def __init__(
        self,
        input_params: FraudLayerParameters,
        hidden_layers: Sequence[FraudLayerParameters],
    ) -> None:
        super().__init__()
        # Classical feature extractor
        self.classical = build_fraud_detection_program(input_params, hidden_layers)

        # Quantum estimator
        self.quantum_estimator = build_quantum_estimator()
        # Expose the circuit and observable for parameter binding
        self.qc = self.quantum_estimator.circuit
        self.observable = self.quantum_estimator.observables[0]

        # Learnable weight parameter for the quantum circuit
        self.q_weight = nn.Parameter(torch.randn(1))

        # Final linear layer to produce scalar output
        self.final = nn.Linear(1, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the hybrid model.

        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of shape (batch, 2) representing two‑dimensional data.

        Returns
        -------
        torch.Tensor
            Regression output of shape (batch, 1).
        """
        # Classical feature extraction
        classical_out = self.classical(inputs).squeeze(-1)  # shape (batch,)

        # Prepare parameter bindings for each sample
        param_bindings = [
            {
                "x": float(val),  # quantum input parameter
                "w": float(self.q_weight),  # quantum weight parameter
            }
            for val in classical_out.tolist()
        ]

        # Evaluate quantum circuit expectation values
        results = self.quantum_estimator.estimator.run(
            circuit=self.qc,
            observables=[self.observable],
            parameter_values=param_bindings,
        ).result

        # Convert results to tensor
        quantum_out = torch.tensor(results, device=inputs.device).unsqueeze(-1)

        # Final linear projection
        return self.final(quantum_out)


__all__ = ["FraudLayerParameters", "build_fraud_detection_program", "HybridEstimator"]
