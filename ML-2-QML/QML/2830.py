"""Hybrid photonic‑Qiskit fraud‑detection program.

The quantum side combines the photonic gates from the original seed with a
parameterised Qiskit circuit that emulates a fully‑connected layer.  The
output of the photonic program is used as a classical feed‑forward value
into the Qiskit circuit, and the expectation value of the qubit is returned
as the final result.  This design allows experimentation with both
continuous‑variable and discrete‑variable quantum hardware.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, Dict

import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate
import qiskit
from qiskit import QuantumCircuit, execute, Aer
import numpy as np


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


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


def _apply_layer(modes: Tuple, params: FraudLayerParameters, *, clip: bool) -> None:
    """Apply a photonic layer to the given modes."""
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


def build_photonic_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> sf.Program:
    """Create the photonic portion of the fraud‑detection program."""
    program = sf.Program(2)
    with program.context as q:
        _apply_layer(q, input_params, clip=False)
        for layer in layers:
            _apply_layer(q, layer, clip=True)
    return program


class QuantumFeatureMapCircuit:
    """A simple Qiskit circuit that mimics a fully‑connected quantum layer."""
    def __init__(self, n_qubits: int = 1, shots: int = 1024) -> None:
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = Aer.get_backend("qasm_simulator")
        self.circuit = QuantumCircuit(n_qubits)
        theta = qiskit.circuit.Parameter("theta")
        self.circuit.h(range(n_qubits))
        self.circuit.barrier()
        self.circuit.ry(theta, range(n_qubits))
        self.circuit.measure_all()
        self.theta = theta

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Execute the parameterised circuit for each theta and return expectation."""
        job = execute(
            self.circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        result = job.result()
        counts = result.get_counts(self.circuit)
        probs = np.array(list(counts.values())) / self.shots
        states = np.array([int(k, 2) for k in counts.keys()], dtype=float)
        expectation = np.sum(states * probs)
        return np.array([expectation])


def build_fraud_detection_quantum_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> Dict[str, object]:
    """Return a hybrid program consisting of a photonic program and a Qiskit circuit.

    The photonic program generates a classical vector that is fed into the
    Qiskit circuit as a set of rotation angles.  The final expectation value
    is returned as the model output.
    """
    photonic_prog = build_photonic_fraud_detection_program(input_params, layers)
    quantum_circ = QuantumFeatureMapCircuit()
    return {"photonic_program": photonic_prog, "quantum_circuit": quantum_circ}


__all__ = ["FraudLayerParameters", "build_fraud_detection_quantum_program", "QuantumFeatureMapCircuit"]
