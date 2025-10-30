"""Hybrid classical‑quantum regressor – ML (PyTorch) implementation."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Iterable, List, Sequence

# Quantum primitives used only for evaluation – kept lightweight
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.primitives import Estimator as StatevectorEstimator


# --------------------------------------------------------------------------- #
#   Fast base estimator – deterministic and noisy evaluation
# --------------------------------------------------------------------------- #
class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrised circuit."""

    def __init__(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[SparsePauliOp],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            results.append([state.expectation_value(obs) for obs in observables])
        return results


class FastEstimator(FastBaseEstimator):
    """Adds optional Gaussian shot noise to the deterministic estimator."""

    def evaluate(
        self,
        observables: Iterable[SparsePauliOp],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in raw:
            noisy_row = [
                rng.normal(mean.real, max(1e-6, 1 / shots))
                + 1j * rng.normal(mean.imag, max(1e-6, 1 / shots))
                for mean in row
            ]
            noisy.append(noisy_row)
        return noisy


# --------------------------------------------------------------------------- #
#   Classical feature extractor – CNN + FC (inspired by Quantum‑NAT)
# --------------------------------------------------------------------------- #
class QFCModel(nn.Module):
    """CNN backbone that projects to four feature maps."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        features = self.features(x)
        flattened = features.view(bsz, -1)
        out = self.fc(flattened)
        return self.norm(out)


# --------------------------------------------------------------------------- #
#   Quantum wrapper – single‑qubit variational layer
# --------------------------------------------------------------------------- #
class QuantumWrapper(nn.Module):
    """Parameterised quantum circuit that maps two scalars to a single expectation value."""

    def __init__(self) -> None:
        super().__init__()
        # Classical trainable parameters – wrapped as PyTorch Parameters
        self.input_param = nn.Parameter(torch.zeros(1))
        self.weight_param = nn.Parameter(torch.zeros(1))

        # Build static circuit template
        self.base_circuit = QuantumCircuit(1)
        self.base_circuit.h(0)
        self.base_circuit.ry(Parameter("θ_input"), 0)
        self.base_circuit.rx(Parameter("θ_weight"), 0)

        # Observable
        self.observable = SparsePauliOp.from_list([("Y", 1)])

        # Estimator for expectation values
        self.estimator = StatevectorEstimator()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Accepts a batch of shape (batch, 2) where
        - inputs[:,0] → input parameter of the circuit
        - inputs[:,1] → weight parameter of the circuit
        Returns a tensor of expectation values of shape (batch, 1).
        """
        batch = inputs.shape[0]
        # Prepare parameter sets
        param_sets = [[float(inputs[i, 0]), float(inputs[i, 1])] for i in range(batch)]
        # Bind parameters and evaluate
        results = self.estimator.run(
            self.base_circuit,
            observables=[self.observable],
            parameter_values=param_sets
        )
        # Convert to torch tensor
        expectations = torch.tensor(
            [res[0].real for res in results], dtype=torch.float32, device=inputs.device
        )
        return expectations.unsqueeze(-1)


# --------------------------------------------------------------------------- #
#   Fraud‑detection program builder (photonic) – for completeness
# --------------------------------------------------------------------------- #
from dataclasses import dataclass
import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate


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


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> sf.Program:
    """Create a Strawberry Fields program for the hybrid fraud detection model."""
    program = sf.Program(2)
    with program.context as q:
        _apply_layer(q, input_params, clip=False)
        for layer in layers:
            _apply_layer(q, layer, clip=True)
    return program


def _apply_layer(modes: Sequence, params: FraudLayerParameters, *, clip: bool) -> None:
    BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
    for i, phase in enumerate(params.phases):
        Rgate(phase) | modes[i]
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        Sgate(r if not clip else _clip(r, 5), phi) | modes[i]
    BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
    for i, phase in enumerate(params.phases):
        Rgate(phase) | modes[i]
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        Dgate(r if not clip else _clip(r, 5), phi) | modes[i]
    for i, k in enumerate(params.kerr):
        Kgate(k if not clip else _clip(k, 1)) | modes[i]


# --------------------------------------------------------------------------- #
#   Hybrid estimator – main model
# --------------------------------------------------------------------------- #
class EstimatorQNN__gen326(nn.Module):
    """
    A hybrid classical‑quantum regression network.
    - CNN backbone (QFCModel) extracts 4 spatial features.
    - QuantumWrapper maps those features to a single expectation value.
    - Final linear layer produces a scalar prediction.
    """

    def __init__(self) -> None:
        super().__init__()
        self.backbone = QFCModel()
        self.quantum = QuantumWrapper()
        self.final = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Forward pass: input shape (batch, 1, 28, 28) – grayscale image.
        Returns predictions of shape (batch, 1).
        """
        features = self.backbone(x)          # (batch, 4)
        quantum_out = self.quantum(features)  # (batch, 1)
        return self.final(quantum_out)

    def evaluate(
        self,
        inputs: torch.Tensor,
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> torch.Tensor:
        """
        Evaluate the network for a batch of inputs with optional shot noise.
        Parameters:
            inputs: (batch, 1, 28, 28) image tensor.
            shots: if None, returns deterministic expectation; otherwise adds Gaussian noise.
            seed: random seed for noise.
        Returns:
            Tensor of predictions (batch, 1).
        """
        self.eval()
        with torch.no_grad():
            preds = self.forward(inputs)
            if shots is not None:
                # Add shot noise to the quantum output before final linear layer
                noise = torch.randn_like(preds) * np.sqrt(1 / shots)
                preds = self.final(preds + noise)
            return preds

__all__ = ["EstimatorQNN__gen326", "build_fraud_detection_program"]
