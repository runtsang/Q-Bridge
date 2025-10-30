"""Quantum module for FraudDetectionHybridNet.

This module implements:
1. A StrawberryFields photonic circuit that mirrors the original `FraudDetection.py` program.
2. A two‑qubit Qiskit variational circuit that accepts a single rotation angle and returns the expectation value of Z on qubit 0.
3. A helper function `hybrid_quantum_photonic_head` that, for each
   sample, first evaluates the photonic circuit (using a Gaussian
   simulator) to obtain an expectation value.  That value is then fed
   as the rotation angle to the Qiskit circuit, and the resulting
   expectation is returned.  The function is vectorised over the
   batch dimension and is fully differentiable when wrapped by
   :class:`torch.autograd.Function` in the ML module.
"""

from __future__ import annotations

import numpy as np
import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate
import qiskit
from qiskit import assemble, transpile
from qiskit.providers.aer import AerSimulator
from dataclasses import dataclass
from typing import Iterable

# ----------------------------------------------------------------------
# Photonic circuit
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

def _apply_layer(modes: sf.Program, params: FraudLayerParameters, *, clip: bool) -> None:
    BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
    for i, phase in enumerate(params.phases):
        Rgate(phase) | modes[i]
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        r_val = _clip(r, 5) if clip else r
        Sgate(r_val, phi) | modes[i]
    BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
    for i, phase in enumerate(params.phases):
        Rgate(phase) | modes[i]
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        r_val = _clip(r, 5) if clip else r
        Dgate(r_val, phi) | modes[i]
    for i, k in enumerate(params.kerr):
        k_val = _clip(k, 1) if clip else k
        Kgate(k_val) | modes[i]

def build_fraud_detection_program(
    params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> sf.Program:
    """Return a StrawberryFields Program for the given layer parameters."""
    prog = sf.Program(2)
    with prog.context as q:
        _apply_layer(q, params, clip=False)
        for layer in layers:
            _apply_layer(q, layer, clip=True)
    return prog

# ----------------------------------------------------------------------
# Quantum (Qiskit) circuit
# ----------------------------------------------------------------------
class QuantumCircuit:
    """Parameterised two‑qubit circuit executed on Aer."""
    def __init__(self, n_qubits: int = 2, shots: int = 1024):
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = AerSimulator()
        self.theta = qiskit.circuit.Parameter("theta")
        self._build_circuit()

    def _build_circuit(self):
        qc = qiskit.QuantumCircuit(self.n_qubits)
        all_qubits = list(range(self.n_qubits))
        qc.h(all_qubits)
        qc.ry(self.theta, all_qubits)
        qc.measure_all()
        self.circuit = qc

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self.circuit, self.backend)
        exp_zs = []
        for theta in thetas:
            qobj = assemble(
                compiled,
                shots=self.shots,
                parameter_binds=[{self.theta: theta}],
            )
            job = self.backend.run(qobj)
            result = job.result()
            counts = result.get_counts()
            # Compute expectation of Z on qubit 0
            prob_0 = sum(counts.get(f"0{b}", 0) for b in ["0", "1"]) / self.shots
            prob_1 = sum(counts.get(f"1{b}", 0) for b in ["0", "1"]) / self.shots
            exp_z = prob_0 - prob_1
            exp_zs.append(exp_z)
        return np.array(exp_zs)

# ----------------------------------------------------------------------
# Hybrid head
# ----------------------------------------------------------------------
def hybrid_quantum_photonic_head(inputs: np.ndarray) -> np.ndarray:
    """
    For each sample in `inputs` (batch, 8) compute:
    1. Photonic expectation via StrawberryFields.
    2. Use that expectation as rotation angle for the Qiskit circuit.
    3. Return the resulting expectation.

    Parameters
    ----------
    inputs : np.ndarray
        Shape (batch, 8).  Each row contains the parameters
        (bs_theta, bs_phi, phases[0], phases[1], squeeze_r[0],
        squeeze_r[1], squeeze_phi[0], squeeze_phi[1]).
        Displacement and Kerr terms are fixed to zero.

    Returns
    -------
    np.ndarray
        Shape (batch,) of quantum expectations.
    """
    batch_size = inputs.shape[0]
    outputs = np.empty(batch_size, dtype=np.float32)
    qc = QuantumCircuit()
    for i in range(batch_size):
        vals = inputs[i]
        params = FraudLayerParameters(
            bs_theta=vals[0],
            bs_phi=vals[1],
            phases=(vals[2], vals[3]),
            squeeze_r=(vals[4], vals[5]),
            squeeze_phi=(vals[6], vals[7]),
            displacement_r=(0.0, 0.0),
            displacement_phi=(0.0, 0.0),
            kerr=(0.0, 0.0),
        )
        prog = build_fraud_detection_program(params, [])
        eng = sf.Engine("gaussian")
        state = eng.run(prog).state
        photon_exp = (
            state.expectation_value(sf.ops.N(0))
            + state.expectation_value(sf.ops.N(1))
        )
        theta = float(photon_exp)
        quantum_exp = qc.run(np.array([theta]))[0]
        outputs[i] = quantum_exp
    return outputs
