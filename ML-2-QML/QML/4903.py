"""
FastHybridEstimator – quantum version.

This module implements the quantum counterpart of the classical
estimator.  It evaluates a Qiskit circuit for a list of parameter
sets and observables, with optional shot sampling and noise.
Convenience constructors for quanvolution, fraud‑detection and
auto‑encoder circuits are also provided.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, List, Any, Union

import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import Statevector, Operator
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RealAmplitudes
from qiskit_machine_learning.neural_networks import SamplerQNN

# ----------------------------------------------------------------------
# Core quantum estimator
# ----------------------------------------------------------------------
class FastHybridEstimator:
    """
    Evaluate a Qiskit circuit for a list of parameter sets and
    observable operators.  Supports optional shot sampling and
    Gaussian noise to emulate measurement error.
    """

    def __init__(self, circuit: QuantumCircuit) -> None:
        self.circuit = circuit
        self.params = list(circuit.parameters)

    def evaluate(
        self,
        observables: Iterable[Operator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """Compute the expectation values for each observable and
        parameter set."""
        observables = list(observables)
        results: List[List[complex]] = []

        if shots is None:
            backend = Aer.get_backend("statevector_simulator")
        else:
            backend = Aer.get_backend("qasm_simulator")

        for values in parameter_sets:
            bound = self.circuit.assign_parameters(
                dict(zip(self.params, values)), inplace=False
            )
            if shots is None:
                state = Statevector.from_instruction(bound)
                row = [state.expectation_value(obs) for obs in observables]
            else:
                job = execute(bound, backend, shots=shots, seed_simulator=seed)
                result = job.result()
                counts = result.get_counts(bound)
                # Build probability distribution from counts
                probs = {
                    tuple(int(bit) for bit in bitstring[::-1]): count / shots
                    for bitstring, count in counts.items()
                }
                row = []
                for obs in observables:
                    exp = 0.0
                    for bitstring, prob in probs.items():
                        exp += prob * obs.data[np.array(bitstring).sum()]
                    row.append(exp)
            results.append(row)

        # Add Gaussian shot noise if requested
        if shots is not None and shots > 0:
            rng = np.random.default_rng(seed)
            noisy = []
            for row in results:
                noisy_row = [
                    complex(rng.normal(float(val.real), 1 / np.sqrt(shots)),
                            rng.normal(float(val.imag), 1 / np.sqrt(shots)))
                    for val in row
                ]
                noisy.append(noisy_row)
            return noisy

        return results


# ----------------------------------------------------------------------
# Quantum helpers
# ----------------------------------------------------------------------
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


def build_fraud_detection_circuit(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> QuantumCircuit:
    """Construct a simple photonic‑style fraud‑detection circuit using
    linear optics primitives (placeholder implementation)."""
    qc = QuantumCircuit(2)
    # Input layer (no clipping)
    _apply_layer(qc, input_params, clip=False)
    for layer in layers:
        _apply_layer(qc, layer, clip=True)
    return qc


def _apply_layer(
    qc: QuantumCircuit, params: FraudLayerParameters, clip: bool
) -> None:
    # Placeholder: use RX/RZ rotations to mimic squeezing/displacement
    qc.rx(params.bs_theta, 0)
    qc.ry(params.bs_phi, 1)
    qc.rz(params.phases[0], 0)
    qc.rz(params.phases[1], 1)
    # Simple linear layer
    qc.cx(0, 1)


# ----------------------------------------------------------------------
# Quanvolution (quantum) helper
# ----------------------------------------------------------------------
def build_quanvolution_circuit() -> QuantumCircuit:
    """A toy 4‑qubit circuit that applies a small parameterized layer
    before measuring all qubits."""
    num_qubits = 4
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector("theta", length=num_qubits)
    for i, p in enumerate(params):
        qc.ry(p, i)
    qc.barrier()
    qc.h(range(num_qubits))
    qc.measure_all()
    return qc


# ----------------------------------------------------------------------
# Quantum auto‑encoder helper
# ----------------------------------------------------------------------
def build_autoencoder_qnn(num_latent: int = 3, num_trash: int = 2) -> SamplerQNN:
    """Return a QNN that implements a simple quantum auto‑encoder using
    a swap‑test style circuit."""
    qr = QuantumCircuit(num_latent + 2 * num_trash + 1, 1)
    # Ansatz
    ansatz = RealAmplitudes(num_latent + num_trash, reps=3)
    qr.compose(ansatz, range(0, num_latent + num_trash), inplace=True)
    qr.barrier()
    aux = num_latent + 2 * num_trash
    qr.h(aux)
    for i in range(num_trash):
        qr.cswap(aux, num_latent + i, num_latent + num_trash + i)
    qr.h(aux)
    qr.measure(aux, 0)

    return SamplerQNN(
        circuit=qr,
        input_params=[],
        weight_params=qr.parameters,
        interpret=lambda x: x,
        output_shape=1,
    )


__all__ = [
    "FastHybridEstimator",
    "FraudLayerParameters",
    "build_fraud_detection_circuit",
    "build_quanvolution_circuit",
    "build_autoencoder_qnn",
]
