"""Quantum side of the hybrid fraud‑detection model.

The module implements two complementary quantum back‑ends:

* a continuous‑variable photonic circuit built with Strawberry Fields
  that mirrors the original photonic implementation, and
* a discrete‑variable variational circuit implemented with Qiskit
  that acts as a fully‑connected layer.

Both back‑ends expose a ``run`` method that returns the state vector
(or expectation value) for a given set of parameters.  The two
implementations can be mixed in a single experiment to study the
advantages of each quantum resource.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate
import qiskit
from qiskit import Aer, execute
from qiskit.circuit import Parameter

__all__ = [
    "FraudLayerParameters",
    "PhotonicFraudCircuit",
    "QuantumFullyConnectedCircuit",
    "FraudDetectionHybrid",
]


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


class PhotonicFraudCircuit:
    """Continuous‑variable photonic circuit for fraud detection."""

    def __init__(self, params: FraudLayerParameters, clip: bool = False, backend: str | None = None) -> None:
        self.params = params
        self.clip = clip
        self.backend = backend or "gaussian_circuit"
        self.program = sf.Program(2)
        with self.program.context as q:
            _apply_layer(q, params, clip=clip)

    def run(self, shots: int = 1024) -> np.ndarray:
        eng = sf.Engine(self.backend, shots=shots)
        result = eng.run(self.program)
        return result.state


class QuantumFullyConnectedCircuit:
    """Discrete‑variable variational circuit that emulates a fully‑connected layer."""

    def __init__(self, n_qubits: int = 1, shots: int = 1024) -> None:
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = Aer.get_backend("qasm_simulator")
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        theta = Parameter("theta")
        self.circuit.h(range(n_qubits))
        self.circuit.barrier()
        self.circuit.ry(theta, range(n_qubits))
        self.circuit.measure_all()

    def run(self, theta: float) -> float:
        bound = self.circuit.bind_parameters({self.circuit.parameters[0]: theta})
        job = execute(bound, self.backend, shots=self.shots)
        result = job.result()
        counts = result.get_counts(bound)
        probs = np.array([counts.get(bit, 0) for bit in sorted(counts)]) / self.shots
        bits = np.array([int(bit, 2) for bit in sorted(counts)])
        expectation = np.sum(bits * probs)
        return expectation


class FraudDetectionHybrid:
    """Hybrid quantum model that bundles a photonic and a Qiskit circuit."""

    def __init__(
        self,
        photonic_params: FraudLayerParameters,
        quantum_params: FraudLayerParameters | None = None,
        n_qubits: int = 1,
    ) -> None:
        self.photonic = PhotonicFraudCircuit(photonic_params, clip=False)
        self.quantum = QuantumFullyConnectedCircuit(n_qubits=n_qubits)

    def run(self, theta: float, shots: int = 1024) -> tuple[np.ndarray, float]:
        photonic_state = self.photonic.run(shots=shots)
        quantum_expect = self.quantum.run(theta)
        return photonic_state, quantum_expect
