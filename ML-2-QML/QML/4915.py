"""Hybrid quantum estimator with utilities for fraud detection, FCL and SamplerQNN."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import List

import numpy as np
import qiskit
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.neural_networks import SamplerQNN as QiskitSamplerQNN
from qiskit.primitives import StatevectorSampler
import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate


# --------------------------------------------------------------------------- #
# Utility helpers
# --------------------------------------------------------------------------- #

def _ensure_batch(values: Sequence[float]) -> Sequence[float]:
    """Return a 1‑D sequence (no-op, kept for API symmetry)."""
    return values


# --------------------------------------------------------------------------- #
# Base estimator
# --------------------------------------------------------------------------- #

class HybridEstimator:
    """Evaluate a pure Qiskit or Strawberry Fields circuit for a batch of parameter sets."""

    def __init__(self, circuit: QuantumCircuit | sf.Program) -> None:
        self.circuit = circuit
        if isinstance(circuit, QuantumCircuit):
            self._parameters = list(circuit.parameters)
        else:  # sf.Program
            self._parameters = [p.name for p in circuit.parameters]

    def evaluate(
        self,
        observables: Iterable,
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Compute expectation values for every parameter set and observable."""
        results: List[List[complex]] = []
        for values in parameter_sets:
            if isinstance(self.circuit, QuantumCircuit):
                bound = self.circuit.assign_parameters(dict(zip(self._parameters, values)))
                state = Statevector.from_instruction(bound)
                row = [state.expectation_value(obs) for obs in observables]
            else:  # Strawberry Fields
                bound_prog = self.circuit.bind_parameters(dict(zip(self._parameters, values)))
                state = sf.Statevector.from_instruction(bound_prog)
                row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results


class HybridEstimatorWithNoise(HybridEstimator):
    """Add Gaussian shot‑noise to a deterministic quantum estimator."""

    def evaluate(
        self,
        observables: Iterable,
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
                complex(
                    rng.normal(x.real, max(1e-6, 1 / shots)),
                    rng.normal(x.imag, max(1e-6, 1 / shots)),
                )
                for x in row
            ]
            noisy.append(noisy_row)
        return noisy


# --------------------------------------------------------------------------- #
# Fraud‑Detection (quantum) utilities
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


def build_fraud_detection_quantum(
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
# Fully‑Connected Layer (quantum) and Sampler QNN (quantum)
# --------------------------------------------------------------------------- #

def FCL() -> QuantumCircuit:
    """Return a simple parameterised Qiskit circuit for a fully‑connected layer."""
    class QuantumCircuitWrapper:
        def __init__(self, n_qubits: int = 1, shots: int = 100) -> None:
            self._circuit = qiskit.QuantumCircuit(n_qubits)
            self.theta = qiskit.circuit.Parameter("theta")
            self._circuit.h(range(n_qubits))
            self._circuit.barrier()
            self._circuit.ry(self.theta, range(n_qubits))
            self._circuit.measure_all()
            self.backend = qiskit.Aer.get_backend("qasm_simulator")
            self.shots = shots

        def run(self, thetas: Iterable[float]) -> np.ndarray:
            job = qiskit.execute(
                self._circuit,
                self.backend,
                shots=self.shots,
                parameter_binds=[{self.theta: theta} for theta in thetas],
            )
            result = job.result().get_counts(self._circuit)
            counts = np.array(list(result.values()))
            states = np.array(list(result.keys())).astype(float)
            probabilities = counts / self.shots
            expectation = np.sum(states * probabilities)
            return np.array([expectation])

    return QuantumCircuitWrapper()


def SamplerQNN() -> QiskitSamplerQNN:
    """Return a Qiskit SamplerQNN instance."""
    inputs2 = ParameterVector("input", 2)
    weights2 = ParameterVector("weight", 4)
    qc2 = QuantumCircuit(2)
    qc2.ry(inputs2[0], 0)
    qc2.ry(inputs2[1], 1)
    qc2.cx(0, 1)
    qc2.ry(weights2[0], 0)
    qc2.ry(weights2[1], 1)
    qc2.cx(0, 1)
    qc2.ry(weights2[2], 0)
    qc2.ry(weights2[3], 1)
    sampler = StatevectorSampler()
    sampler_qnn = QiskitSamplerQNN(
        circuit=qc2, input_params=inputs2, weight_params=weights2, sampler=sampler
    )
    return sampler_qnn


__all__ = [
    "HybridEstimator",
    "HybridEstimatorWithNoise",
    "build_fraud_detection_quantum",
    "FCL",
    "SamplerQNN",
    "FraudLayerParameters",
]
