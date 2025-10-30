"""Unified fraud detection estimator for quantum backends."""

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from typing import Iterable, List, Sequence, Tuple
from dataclasses import dataclass


# --- Parameter container ----------------------------------------------------- #
@dataclass
class FraudLayerParameters:
    """Parameters describing a single photonic layer."""
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]


# --- Utility helpers -------------------------------------------------------- #
def _clip(value: float, bound: float) -> float:
    return max(-bound, min(value, bound))


def _apply_layer(qc: QuantumCircuit, params: FraudLayerParameters, *, clip: bool) -> None:
    """Add a photonic‑like layer to a Qiskit circuit."""
    # Beam splitter (mode‑mixing) is emulated by a CX gate for two qubits
    qc.cx(0, 1)
    for i, phase in enumerate(params.phases):
        qc.rz(phase, i)
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        qc.s(i) if clip else qc.id(i)  # placeholder for squeezing
    qc.cx(0, 1)
    for i, phase in enumerate(params.phases):
        qc.rz(phase, i)
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        qc.rx(r, i)  # placeholder for displacement
    for i, k in enumerate(params.kerr):
        qc.rxx(k, 0, 1)  # placeholder for Kerr nonlinearity


# ----- Quantum fraud program builder ---------------------------------------- #
def build_quantum_fraud_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> QuantumCircuit:
    """Create a Qiskit circuit that mirrors the photonic architecture."""
    qc = QuantumCircuit(2)
    _apply_layer(qc, input_params, clip=False)
    for layer in layers:
        _apply_layer(qc, layer, clip=True)
    return qc


# ----- Unified estimator ----------------------------------------------------- #
class UnifiedFraudEstimator:
    """Evaluate deterministic or noisy expectation values for a parameterised quantum circuit."""

    def __init__(self, circuit: QuantumCircuit):
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, param_values: Sequence[float]) -> QuantumCircuit:
        if len(param_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, param_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Compute expectation values for every parameter set and observable."""
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

    def evaluate_with_noise(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int = 1_000,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """Add Gaussian shot‑noise to deterministic expectation values."""
        raw = self.evaluate(observables, parameter_sets)
        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in raw:
            noisy_row = [
                complex(rng.normal(val.real, max(1e-6, 1 / shots)),
                        rng.normal(val.imag, max(1e-6, 1 / shots)))
                for val in row
            ]
            noisy.append(noisy_row)
        return noisy

    @staticmethod
    def build_quantum_fraud_program(
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
    ) -> QuantumCircuit:
        """Convenience wrapper that returns a fully‑constructed Qiskit circuit."""
        return build_quantum_fraud_program(input_params, layers)


__all__ = ["FraudLayerParameters", "build_quantum_fraud_program", "UnifiedFraudEstimator"]
