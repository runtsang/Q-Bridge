from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import numpy as np
import qiskit
from qiskit import QuantumCircuit, AerSimulator

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

def _apply_layer(circ: QuantumCircuit, params: FraudLayerParameters, *, clip: bool) -> None:
    # Beam‑splitter analogue: two RZ rotations followed by RX
    circ.rz(params.bs_theta, 0)
    circ.rz(params.bs_phi, 1)
    circ.rx(params.bs_theta, 0)
    circ.rx(params.bs_phi, 1)
    # Phases
    for i, phase in enumerate(params.phases):
        circ.rz(phase, i)
    # Squeezing analogue with RX and RZ
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        circ.rx(_clip(r, 5) if clip else r, i)
        circ.rz(_clip(phi, 5) if clip else phi, i)
    # Displacement analogue with RY and RZ
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        circ.ry(_clip(r, 5) if clip else r, i)
        circ.rz(_clip(phi, 5) if clip else phi, i)
    # Kerr analogue with RZ
    for i, k in enumerate(params.kerr):
        circ.rz(_clip(k, 1) if clip else k, i)
    circ.barrier()

class QuantumFCL:
    """Parameterised single‑qubit circuit that mimics the classical FCL."""
    def __init__(self, backend: qiskit.providers.Backend, shots: int = 1024):
        self.backend = backend
        self.shots = shots
        self.circ = QuantumCircuit(1, 1)
        self.theta = qiskit.circuit.Parameter("theta")
        self.circ.h(0)
        self.circ.ry(self.theta, 0)
        self.circ.measure(0, 0)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        jobs = []
        for theta in thetas:
            bound = {self.theta: theta}
            bound_circ = self.circ.bind_parameters(bound)
            job = self.backend.run(bound_circ, shots=self.shots)
            jobs.append(job)
        results = []
        for job in jobs:
            counts = job.result().get_counts()
            probs = np.array(list(counts.values())) / self.shots
            states = np.array([int(k) for k in counts.keys()])
            expectation = np.sum(states * probs)
            results.append(expectation)
        return np.array(results)

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
    fcl_layers: int = 0,
) -> list[QuantumCircuit]:
    """Return a list of quantum circuits that emulate the classical fraud‑detection stack."""
    circuits = []
    backend = AerSimulator()
    # Main photonic‑style circuit
    main_circ = QuantumCircuit(2, 2)
    _apply_layer(main_circ, input_params, clip=False)
    for layer in layers:
        _apply_layer(main_circ, layer, clip=True)
    main_circ.barrier()
    main_circ.measure([0, 1], [0, 1])
    circuits.append(main_circ)
    # Append fully‑connected quantum layers
    for _ in range(fcl_layers):
        fcl = QuantumFCL(backend)
        circuits.append(fcl.circ)
    return circuits

__all__ = ["FraudLayerParameters", "build_fraud_detection_program", "QuantumFCL"]
