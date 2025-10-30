"""Hybrid fraud detection quantum program.

Provides a Strawberry Fields program for the photonic part and a
Qiskit circuit for the fully‑connected quantum sub‑layer.  The module
exposes a single build function that returns a dictionary containing
both programs.  The user can then execute the photonic program on a
continuous‑variable backend and the quantum circuits on a discrete‑qubit
backend and fuse the results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Dict, List

import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate

import qiskit
from qiskit import QuantumCircuit, Aer, execute
import numpy as np

@dataclass
class PhotonicLayerParameters:
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

def _apply_layer(modes: List, params: PhotonicLayerParameters, *, clip: bool) -> None:
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

def _build_fcl_circuit(n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))
    qc.barrier()
    theta = qiskit.circuit.Parameter("theta")
    for i in range(n_qubits):
        qc.ry(theta, i)
    qc.measure_all()
    return qc

def build_fraud_detection_program(
    input_params: PhotonicLayerParameters,
    layers: Iterable[PhotonicLayerParameters],
    n_qubits: int = 2,
) -> Dict[str, any]:
    """Construct both the photonic and quantum parts of the hybrid model."""
    # Photonic part
    photonic_prog = sf.Program(2)
    with photonic_prog.context as q:
        _apply_layer(q, input_params, clip=False)
        for layer in layers:
            _apply_layer(q, layer, clip=True)
    # Quantum part
    quantum_circuits = [_build_fcl_circuit(n_qubits) for _ in layers]
    return {"photonic": photonic_prog, "quantum": quantum_circuits}

def run_hybrid(
    programs: Dict[str, any],
    photonic_input: np.ndarray | None = None,
    quantum_thetas: List[np.ndarray] | None = None,
    shots: int = 1024,
) -> Dict[str, np.ndarray]:
    """Execute both photonic and quantum components and return raw results."""
    results = {}
    # Photonic simulation
    photonic_prog = programs["photonic"]
    if photonic_input is not None:
        photonic_prog.preprocess.set_state(photonic_input)
    sim = sf.backends.Simulator("gaussian")
    result = sim.run(photonic_prog).state
    results["photonic"] = result
    # Quantum simulation
    quantum_circuits = programs["quantum"]
    backend = Aer.get_backend("qasm_simulator")
    for idx, qc in enumerate(quantum_circuits):
        theta_val = quantum_thetas[idx] if quantum_thetas else np.ones_like(qc.parameters)
        bound_params = {qc.parameters[0]: theta_val}
        job = execute(qc, backend=backend, shots=shots, parameter_binds=[bound_params])
        counts = job.result().get_counts(qc)
        # convert counts to expectation of Z
        probs = np.array(list(counts.values())) / shots
        states = np.array([int(k, 2) for k in counts.keys()])
        expectation = np.sum(states * probs)
        results[f"quantum_{idx}"] = np.array([expectation])
    return results

__all__ = [
    "PhotonicLayerParameters",
    "build_fraud_detection_program",
    "run_hybrid",
]
