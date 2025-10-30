"""Hybrid quantum layer that extends the simple fully‑connected circuit with photonic‑style parameters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter


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


class HybridLayer:
    """Quantum circuit that emulates a fully‑connected layer with photonic‑style controls."""

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        backend=None,
        shots: int = 1024,
    ) -> None:
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.theta_params = [Parameter(f"theta_{i}") for i in range(2)]
        self.circuit = self._build_circuit(input_params, layers)

    def _build_circuit(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
    ) -> QuantumCircuit:
        qc = QuantumCircuit(2)
        # Input layer
        self._apply_layer(qc, input_params, clip=False)
        # Hidden layers
        for layer in layers:
            self._apply_layer(qc, layer, clip=True)
        # Parameterized rotation per qubit
        for i, theta in enumerate(self.theta_params):
            qc.ry(theta, i)
        qc.measure_all()
        return qc

    def _apply_layer(self, qc: QuantumCircuit, params: FraudLayerParameters, *, clip: bool) -> None:
        # Beam splitter emulation via Hadamard and rotations
        theta = params.bs_theta if not clip else _clip(params.bs_theta, 5)
        phi = params.bs_phi if not clip else _clip(params.bs_phi, 5)
        qc.h(0)
        qc.h(1)
        for qubit in range(2):
            qc.ry(theta, qubit)
            qc.rz(phi, qubit)
        # Phase gates
        for qubit, phase in enumerate(params.phases):
            qc.rz(phase, qubit)
        # Squeezing via additional rotations
        for qubit, (r, p) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
            qc.ry(r if not clip else _clip(r, 5), qubit)
            qc.rz(p, qubit)
        # Displacement via rotation
        for qubit, (r, p) in enumerate(zip(params.displacement_r, params.displacement_phi)):
            qc.rz(r if not clip else _clip(r, 5), qubit)
            qc.ry(p, qubit)
        # Kerr effect via controlled phase
        for qubit, k in enumerate(params.kerr):
            qc.cu1(k if not clip else _clip(k, 1), qubit, (qubit + 1) % 2)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Execute the circuit with the supplied parameters and return the expectation value."""
        param_bindings = [{self.theta_params[i]: t for i, t in enumerate(thetas)}]
        job = execute(self.circuit, self.backend, shots=self.shots, parameter_binds=param_bindings)
        result = job.result()
        counts = result.get_counts(self.circuit)
        probs = np.array(list(counts.values())) / self.shots
        states = np.array([int(k, 2) for k in counts.keys()], dtype=float)
        expectation = np.sum(states * probs)
        return np.array([expectation])


def FCL() -> HybridLayer:
    """Return an instance of the hybrid quantum layer."""
    input_params = FraudLayerParameters(
        bs_theta=0.1,
        bs_phi=0.2,
        phases=(0.3, 0.4),
        squeeze_r=(0.5, 0.6),
        squeeze_phi=(0.7, 0.8),
        displacement_r=(0.9, 1.0),
        displacement_phi=(1.1, 1.2),
        kerr=(1.3, 1.4),
    )
    layers = [
        FraudLayerParameters(
            bs_theta=0.2,
            bs_phi=0.3,
            phases=(0.4, 0.5),
            squeeze_r=(0.6, 0.7),
            squeeze_phi=(0.8, 0.9),
            displacement_r=(1.0, 1.1),
            displacement_phi=(1.2, 1.3),
            kerr=(1.4, 1.5),
        )
    ]
    return HybridLayer(input_params, layers)


__all__ = ["HybridLayer", "FCL", "FraudLayerParameters"]
