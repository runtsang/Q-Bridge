"""Hybrid classical‑quantum regressor – QML (Qiskit) implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate

from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator

# --------------------------------------------------------------------------- #
#   Photonic fraud‑detection program builder (kept identical to ML side)
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


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> sf.Program:
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
#   Quantum EstimatorQNN – parameterised circuit
# --------------------------------------------------------------------------- #
class EstimatorQNN__gen326:
    """
    Qiskit implementation of the hybrid estimator.
    The circuit mirrors the QuantumWrapper used in the PyTorch side.
    """

    def __init__(self) -> None:
        # Parameters
        self.input_param = Parameter("θ_input")
        self.weight_param = Parameter("θ_weight")

        # Base circuit
        self.circuit = QuantumCircuit(1)
        self.circuit.h(0)
        self.circuit.ry(self.input_param, 0)
        self.circuit.rx(self.weight_param, 0)

        # Observable
        self.observable = SparsePauliOp.from_list([("Y", 1)])

        # Estimator backend
        self.estimator = StatevectorEstimator()

        # Wrap as EstimatorQNN for compatibility
        self.estimator_qnn = EstimatorQNN(
            circuit=self.circuit,
            observables=self.observable,
            input_params=[self.input_param],
            weight_params=[self.weight_param],
            estimator=self.estimator,
        )

    def evaluate(
        self,
        inputs: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> Sequence[Sequence[complex]]:
        """
        Evaluate the quantum circuit for a list of parameter sets.
        Parameters:
            inputs: iterable of [input, weight] pairs.
            shots: if None, deterministic; otherwise emulates shot noise.
            seed: random seed for noise.
        Returns:
            Nested list of expectation values.
        """
        if shots is None:
            results = self.estimator_qnn.evaluate(
                inputs=inputs,
                observables=[self.observable],
            )
        else:
            # Use classical noise injection – mimic FastEstimator
            rng = np.random.default_rng(seed)
            deterministic = self.estimator_qnn.evaluate(
                inputs=inputs,
                observables=[self.observable],
            )
            results = []
            for row in deterministic:
                noisy_row = [
                    rng.normal(val.real, max(1e-6, 1 / shots))
                    + 1j * rng.normal(val.imag, max(1e-6, 1 / shots))
                    for val in row
                ]
                results.append(noisy_row)
        return results


__all__ = ["EstimatorQNN__gen326", "build_fraud_detection_program"]
