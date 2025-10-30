from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from typing import Iterable, Sequence, List, Callable

# ----------------------------------------------------------------------
#  Data generation (identical to the classical seed)
# ----------------------------------------------------------------------
def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic superposition states and labels."""
    omega_0 = np.zeros(2 ** num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_wires, dtype=complex)
    omega_1[-1] = 1.0
    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels

# ----------------------------------------------------------------------
#  Photonic‑style classical layer parameters (kept for consistency)
# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
#  Build a parameterised quantum circuit that mirrors the photonic layers
# ----------------------------------------------------------------------
def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> QuantumCircuit:
    qc = QuantumCircuit(2)
    def _apply_layer(qc: QuantumCircuit, params: FraudLayerParameters, clip: bool) -> None:
        # First set of rotations (BS‑like)
        qc.ry(params.bs_theta, 0)
        qc.ry(params.bs_phi, 1)
        # Phase rotations
        for i, phase in enumerate(params.phases):
            qc.rz(phase, i)
        # Squeezing (approximated with Rx)
        for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
            qc.rx(_clip(r, 5), i)
        # Second set of rotations
        qc.ry(params.bs_theta, 0)
        qc.ry(params.bs_phi, 1)
        # Phase rotations again
        for i, phase in enumerate(params.phases):
            qc.rz(phase, i)
        # Displacement (approximated with Rx)
        for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
            qc.rx(_clip(r, 5), i)
        # Kerr (approximated with Rz)
        for i, k in enumerate(params.kerr):
            qc.rz(_clip(k, 1), i)
    _apply_layer(qc, input_params, clip=False)
    for layer in layers:
        _apply_layer(qc, layer, clip=True)
    return qc

# ----------------------------------------------------------------------
#  Fast estimator for Qiskit circuits
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

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

# ----------------------------------------------------------------------
#  Quantum‑only hybrid fraud‑detection model
# ----------------------------------------------------------------------
class FraudDetectionHybrid:
    """Quantum‑only fraud‑detection circuit with a classical post‑processing head.

    The circuit is a stack of photonic‑style layers (rotations, squeezers, displacements,
    Kerr) and a final measurement of Pauli‑Z.  A linear layer is applied to the
    expectation value to produce the scalar output.  The model can be used
    directly with a Qiskit StatevectorEstimator or a custom FastBaseEstimator.
    """
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Sequence[FraudLayerParameters],
        *,
        backend: str = "qasm_simulator",
    ) -> None:
        self.circuit = build_fraud_detection_program(input_params, layers)
        self.backend = backend
        self.head = nn.Linear(1, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Encode each input sample as a set of circuit parameters
        # Here we use the first two input dimensions as rotation angles
        param_sets = inputs.detach().cpu().numpy().tolist()
        estimator = FastBaseEstimator(self.circuit)
        # Measure Pauli‑Z on qubit 0 for each sample
        pauli_z = Statevector.from_instruction(self.circuit).pauli_expectation_value("Z", 0)
        # Since FastBaseEstimator expects a list of observables, we wrap the measurement
        row = estimator.evaluate([pauli_z], param_sets)
        # Convert list of lists to tensor and apply linear head
        exp_vals = torch.tensor([r[0] for r in row], dtype=torch.float32).unsqueeze(-1)
        return self.head(exp_vals)

__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "generate_superposition_data",
    "FastBaseEstimator",
    "FraudDetectionHybrid",
]
