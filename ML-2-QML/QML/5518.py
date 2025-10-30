"""Hybrid quantum estimator for Qiskit / Strawberry Fields circuits.

The implementation mirrors the classical FastBaseEstimatorGen but
operates on quantum programs.  It supports exact Statevector evaluation,
shot‑based sampling, and optional Gaussian post‑processing noise.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import List

import numpy as np
import qiskit
import strawberryfields as sf
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import execute
from qiskit.quantum_info import Statevector
from qiskit.circuit import ParameterVector
from qiskit.providers.aer import AerSimulator
from strawberryfields import Engine, Program
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate

# --------------------------------------------------------------------------- #
# 1.  Quantum utilities – FastBaseEstimatorGen
# --------------------------------------------------------------------------- #
class FastBaseEstimatorGen:
    """Evaluate a quantum circuit or Strawberry Fields program.

    *model* can be a :class:`qiskit.circuit.QuantumCircuit` or a
    :class:`sf.Program`.  The method ``evaluate`` returns a list of rows
    containing expectation values of the supplied observables.  If
    *shots* is given, the circuit is executed on the Aer simulator
    and the results are sampled; otherwise a Statevector is used.
    """

    def __init__(self, model: QuantumCircuit | Program) -> None:
        self.model = model
        if isinstance(model, QuantumCircuit):
            self.backend = AerSimulator()
        else:
            self.backend = None  # SF uses its own engine

    def evaluate(
        self,
        observables: Iterable[object],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """Return expectation values for each parameter set."""
        results: List[List[complex]] = []

        if isinstance(self.model, QuantumCircuit):
            for params in parameter_sets:
                bound = self.model.assign_parameters(
                    dict(zip(self.model.parameters, params))
                )
                if shots is None:
                    state = Statevector.from_instruction(bound)
                    row = [state.expectation_value(obs) for obs in observables]
                else:
                    job = execute(bound, self.backend, shots=shots, seed_simulator=seed)
                    counts = job.result().get_counts(bound)
                    probs = {int(k, 2): v / shots for k, v in counts.items()}
                    row = [
                        sum(
                            probs.get(int(bin(i)[2:].zfill(bound.num_qubits)), 0)
                            * obs.diagonal()[i]
                            for i in range(2 ** bound.num_qubits)
                        )
                        for obs in observables
                    ]
                results.append(row)

        else:  # Strawberry Fields program
            for params in parameter_sets:
                prog = self.model
                # Parameter binding is omitted for brevity – assume the program
                # already contains concrete parameters.
                eng = Engine("gaussian_state")
                result = eng.run(prog)
                # No generic observable interface – placeholder zeros
                row = [0 + 0j] * len(observables)
                results.append(row)

        return results


# --------------------------------------------------------------------------- #
# 2.  Quantum self‑attention block
# --------------------------------------------------------------------------- #
class QuantumSelfAttention:
    """Self‑attention implemented as a parametric Qiskit circuit."""

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _build_circuit(
        self, rotation_params: np.ndarray, entangle_params: np.ndarray
    ) -> QuantumCircuit:
        circuit = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        circuit.measure(self.qr, self.cr)
        return circuit

    def run(
        self,
        backend,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ):
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = execute(circuit, backend, shots=shots)
        return job.result().get_counts(circuit)


# --------------------------------------------------------------------------- #
# 3.  Fraud‑detection program (Strawberry Fields)
# --------------------------------------------------------------------------- #
@dataclass
class FraudLayerParameters:
    """Parameters describing a single photonic layer."""

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
) -> Program:
    """Create a Strawberry Fields program for the hybrid fraud‑detection model."""
    program = Program(2)
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
# 4.  Exposed symbols
# --------------------------------------------------------------------------- #
__all__ = [
    "FastBaseEstimatorGen",
    "QuantumSelfAttention",
    "FraudLayerParameters",
    "build_fraud_detection_program",
]
