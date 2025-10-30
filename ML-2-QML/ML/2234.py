"""Hybrid fraud detection model combining classical neural layers and a quantum classifier.

This module extends the classical FraudDetection architecture by optionally attaching a quantum
variational circuit.  The classical part mirrors the photonic layers defined in the seed,
while the quantum part follows the Qiskit implementation from QuantumClassifierModel.py.
The two halves reinforce each other: classical preprocessing reduces the input space, and the
quantum circuit provides a richer, non‑linear decision boundary.

The class is fully PyTorch‑compatible: it can be used as a drop‑in replacement for the
original FraudDetection model, and the quantum part can be swapped in by passing a
callable that evaluates a quantum circuit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Callable, Optional, Tuple

import torch
from torch import nn

# Re‑use the FraudLayerParameters dataclass from the original seed
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

class HybridFraudClassifier(nn.Module):
    """
    Hybrid fraud detection model.

    Parameters
    ----------
    input_params : FraudLayerParameters
        Parameters for the first (input) layer.
    layers : Iterable[FraudLayerParameters]
        Parameters for the subsequent layers.
    quantum_circuit : Callable[[torch.Tensor], torch.Tensor] | None
        Optional callable that evaluates a quantum circuit on the output of the
        classical network.  If ``None`` the model reduces to the pure classical
        architecture.
    """

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        quantum_circuit: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        self.classical = build_fraud_detection_program(input_params, layers)
        self.quantum_circuit = quantum_circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.classical(x)
        if self.quantum_circuit is not None:
            out = self.quantum_circuit(out)
        return out

    @staticmethod
    def build_quantum_classifier(num_qubits: int, depth: int) -> Tuple[Callable[[torch.Tensor], torch.Tensor], Iterable, Iterable, list]:
        """
        Helper to construct a quantum circuit compatible with this hybrid model.

        Returns a tuple containing:
        * a callable that evaluates the circuit
        * encoding parameters
        * variational parameters
        * measurement observables
        """
        # Import lazily to keep the module free of quantum dependencies
        from.qml import build_classifier_circuit  # type: ignore
        circuit, encoding, weights, observables = build_classifier_circuit(num_qubits, depth)
        return circuit, encoding, weights, observables

__all__ = ["FraudLayerParameters", "HybridFraudClassifier", "build_fraud_detection_program"]
