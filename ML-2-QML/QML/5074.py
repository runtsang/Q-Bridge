"""
Quantum hybrid fraud‑detection module using Qiskit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import StatevectorSampler

# --------------------------------------------------------------------------- #
# 1. Parameters describing a photonic‑like layer
# --------------------------------------------------------------------------- #
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

# --------------------------------------------------------------------------- #
# 2. Helper: clip values
# --------------------------------------------------------------------------- #
def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

# --------------------------------------------------------------------------- #
# 3. Build a single layer as a sub‑circuit
# --------------------------------------------------------------------------- #
def _apply_layer(qc: QuantumCircuit, params: FraudLayerParameters, *, clip: bool) -> None:
    # Beam‑splitter analogue – RX + RZ on two qubits
    qc.rx(params.bs_theta, 0)
    qc.rz(params.bs_phi, 1)

    # Phase gates
    qc.rz(params.phases[0], 0)
    qc.rz(params.phases[1], 1)

    # Squeezing analogue – use RY with clipped parameters
    qc.ry(_clip(params.squeeze_r[0], 5.0) if clip else params.squeeze_r[0], 0)
    qc.ry(_clip(params.squeeze_r[1], 5.0) if clip else params.squeeze_r[1], 1)

    # Displacement analogue – use RX with clipped parameters
    qc.rx(_clip(params.displacement_r[0], 5.0) if clip else params.displacement_r[0], 0)
    qc.rx(_clip(params.displacement_r[1], 5.0) if clip else params.displacement_r[1], 1)

    # Kerr analogue – use RZ with clipped parameters
    qc.rz(_clip(params.kerr[0], 1.0) if clip else params.kerr[0], 0)
    qc.rz(_clip(params.kerr[1], 1.0) if clip else params.kerr[1], 1)

# --------------------------------------------------------------------------- #
# 4. Build entire fraud‑detection circuit
# --------------------------------------------------------------------------- #
def build_quantum_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> QuantumCircuit:
    """Create a Qiskit circuit mirroring the photonic fraud‑detection stack."""
    qr = QuantumRegister(2, "q")
    cr = ClassicalRegister(2, "c")
    qc = QuantumCircuit(qr, cr)

    _apply_layer(qc, input_params, clip=False)
    for l in layers:
        _apply_layer(qc, l, clip=True)

    # Final measurement
    qc.measure(qr, cr)
    return qc

# --------------------------------------------------------------------------- #
# 5. Hybrid quantum model
# --------------------------------------------------------------------------- #
class FraudDetectionHybrid:
    """
    Quantum hybrid fraud‑detection model.

    Uses a parameterised Qiskit circuit (built from :class:`FraudLayerParameters`)
    and a Qiskit SamplerQNN to obtain probability distributions.  The
    StatevectorSampler is employed to evaluate a quantum kernel between
    the model state and a set of reference states.
    """

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        backend=None,
    ) -> None:
        self.qc = build_quantum_fraud_detection_program(input_params, layers)
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.sampler = SamplerQNN()
        # Prepare a StatevectorSampler for kernel evaluation
        self.state_sampler = StatevectorSampler()

    def predict(self, shots: int = 1024) -> dict[str, int]:
        """Run the circuit and return measurement counts."""
        job = execute(self.qc, self.backend, shots=shots)
        return job.result().get_counts(self.qc)

    def kernel_score(self, reference_states: Iterable[np.ndarray]) -> np.ndarray:
        """
        Compute a quantum kernel between the current circuit state and a list
        of reference states (each a 4‑dimensional amplitude vector).

        Parameters
        ----------
        reference_states
            List of classical state vectors (length 4, normalized).
        """
        # Obtain statevector of the current circuit
        sv = self.state_sampler.run(self.qc, shots=1).statevector

        # Normalize just in case
        sv = sv / np.linalg.norm(sv)

        # Compute inner products with reference states
        return np.array([abs(np.vdot(sv, ref))**2 for ref in reference_states])

__all__ = [
    "FraudLayerParameters",
    "build_quantum_fraud_detection_program",
    "FraudDetectionHybrid",
]
