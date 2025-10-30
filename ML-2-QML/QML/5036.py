"""Quantum implementation of the fraud‑detection program.

The module defines a single ``FraudDetectionHybrid`` class that
mirrors the classical architecture:
- A sequence of parameterised layers defined by ``FraudLayerParameters``.
- An optional auto‑encoder style ansatz (via ``RealAmplitudes``).
- An expectation value measurement that can be used as a scalar output.

The class exposes a ``evaluate`` method compatible with the classical
``FastEstimator`` API, returning expectation values for any set of
observables.
"""

from __future__ import annotations

from typing import Iterable, Sequence, List
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.circuit.library import RealAmplitudes
from qiskit.circuit import Parameter
import numpy as np

# ----------------------------------------------------------------------
# Fraud‑detection layer definition (from FraudDetection.py seed)
# ----------------------------------------------------------------------
class FraudLayerParameters:
    def __init__(self, bs_theta, bs_phi, phases, squeeze_r, squeeze_phi, displacement_r, displacement_phi, kerr):
        self.bs_theta = bs_theta
        self.bs_phi = bs_phi
        self.phases = phases
        self.squeeze_r = squeeze_r
        self.squeeze_phi = squeeze_phi
        self.displacement_r = displacement_r
        self.displacement_phi = displacement_phi
        self.kerr = kerr

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

# ----------------------------------------------------------------------
# FastBaseEstimator (from FastBaseEstimator.py seed)
# ----------------------------------------------------------------------
class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrised circuit."""
    def __init__(self, circuit: QuantumCircuit):
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(self, observables: Iterable[BaseOperator], parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

# ----------------------------------------------------------------------
# Hybrid fraud‑detection quantum circuit
# ----------------------------------------------------------------------
class FraudDetectionHybrid:
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        autoencoder_depth: int = 3,
        backend: str = "qiskit",
        shots: int | None = None,
    ) -> None:
        self.shots = shots
        self.circuit = self._build_program(input_params, layers, autoencoder_depth)
        self.estimator = FastBaseEstimator(self.circuit)

    def _build_program(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        autoencoder_depth: int,
    ) -> QuantumCircuit:
        circuit = QuantumCircuit()
        # Input layer
        self._apply_layer(circuit, input_params, clip=False)
        # Subsequent layers
        for layer in layers:
            self._apply_layer(circuit, layer, clip=True)
        # Auto‑encoder ansatz
        self._apply_autoencoder(circuit, autoencoder_depth)
        # Measurement
        circuit.measure_all()
        return circuit

    def _apply_layer(self, circuit: QuantumCircuit, params: FraudLayerParameters, *, clip: bool) -> None:
        # Beam‑splitter equivalent: two RY gates and a CNOT
        circuit.ry(params.bs_theta, 0)
        circuit.ry(params.bs_phi, 1)
        circuit.cx(0, 1)
        # Phase gates
        circuit.rz(params.phases[0], 0)
        circuit.rz(params.phases[1], 1)
        # Squeezing / displacement approximated by rotation
        for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
            val = r if not clip else _clip(r, 5)
            circuit.ry(val, i % 2)
            circuit.rz(phi, i % 2)
        # Kerr nonlinearity approximated by RZZ
        for i, k in enumerate(params.kerr):
            val = k if not clip else _clip(k, 1)
            circuit.rzz(val, 0, 1)

    def _apply_autoencoder(self, circuit: QuantumCircuit, depth: int) -> None:
        # Use a RealAmplitudes ansatz per qubit
        ansatz = RealAmplitudes(num_qubits=depth, reps=3)
        circuit.compose(ansatz, inplace=True)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Return expectation values for each observable and parameter set."""
        return self.estimator.evaluate(observables, parameter_sets)

__all__ = ["FraudDetectionHybrid"]
