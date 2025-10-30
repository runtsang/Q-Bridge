from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit import ParameterVector
from qiskit.algorithms import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN as QiskitSamplerQNN


# --------------------------------------------------------------------------- #
# Quantum counterparts of the classical fraud‑detection primitives
# --------------------------------------------------------------------------- #

@dataclass
class FraudLayerParameters:
    """Parameters describing a single photonic layer (used for quantum bounds)."""
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
    layers: Iterable[FraudLayerParameters]
) -> QuantumCircuit:
    """
    Build a Qiskit circuit that mirrors the photonic fraud‑detection layers.
    The circuit uses parameterized RY gates for inputs, CX gates for entanglement,
    and bounded parameters to emulate squeezing/displacement limits.
    """
    qc = QuantumCircuit(2)
    # Input rotations
    qc.ry(input_params.bs_theta, 0)
    qc.ry(input_params.bs_phi, 1)
    # Entangling block
    qc.cx(0, 1)
    # Phase operations (simulated by RZ)
    for i, phase in enumerate(input_params.phases):
        qc.rz(phase, i)
    # Bounding to emulate squeezing limits
    for i, (r, phi) in enumerate(zip(input_params.squeeze_r, input_params.squeeze_phi)):
        qc.rz(_clip(r, 5.0), i)
    # Second entangling
    qc.cx(0, 1)
    # Phase again
    for i, phase in enumerate(input_params.phases):
        qc.rz(phase, i)
    # Displacement via RY with clipping
    for i, (r, phi) in enumerate(zip(input_params.displacement_r, input_params.displacement_phi)):
        qc.ry(_clip(r, 5.0), i)
    # Kerr‑like nonlinearity (simulated by RZ with small bound)
    for i, k in enumerate(input_params.kerr):
        qc.rz(_clip(k, 1.0), i)
    # Final measurement
    qc.measure_all()
    return qc


# --------------------------------------------------------------------------- #
# SamplerQNN equivalent: parameterized circuit + Qiskit Sampler
# --------------------------------------------------------------------------- #

def build_sampler_qnn_circuit() -> QuantumCircuit:
    """
    Create a Qiskit circuit that matches the SamplerQNN example:
    two input parameters and four weight parameters.
    """
    inputs = ParameterVector("input", 2)
    weights = ParameterVector("weight", 4)
    qc = QuantumCircuit(2)
    qc.ry(inputs[0], 0)
    qc.ry(inputs[1], 1)
    qc.cx(0, 1)
    qc.ry(weights[0], 0)
    qc.ry(weights[1], 1)
    qc.cx(0, 1)
    qc.ry(weights[2], 0)
    qc.ry(weights[3], 1)
    qc.measure_all()
    return qc


def sampler_qnn(
    input_vals: Sequence[float],
    weight_vals: Sequence[float],
    shots: int = 1024
) -> np.ndarray:
    """
    Execute the SamplerQNN circuit on a Qiskit simulator and return the
    probability distribution over the two‑qubit basis states.
    """
    qc = build_sampler_qnn_circuit()
    param_bindings = [
        {qc.parameters[i]: val for i, val in enumerate(input_vals + weight_vals)}
    ]
    simulator = Aer.get_backend("aer_simulator")
    job = execute(qc, simulator, shots=shots, parameter_binds=param_bindings)
    result = job.result()
    counts = result.get_counts(qc)
    probs = np.array([counts.get(bitstring, 0) / shots for bitstring in sorted(counts)])
    return probs


# --------------------------------------------------------------------------- #
# FCL‑style expectation circuit
# --------------------------------------------------------------------------- #

def build_fcl_circuit() -> QuantumCircuit:
    """
    Build a single‑qubit circuit that outputs an expectation value
    after a parameterized rotation. This mimics the FCL example.
    """
    qc = QuantumCircuit(1)
    qc.h(0)
    theta = ParameterVector("theta", 1)[0]
    qc.ry(theta, 0)
    qc.measure_all()
    return qc


def expectation_from_fcl(theta_vals: Sequence[float], shots: int = 1024) -> np.ndarray:
    """
    Run the FCL circuit and compute the expectation value of the measured
    qubit (treated as 0/1). Returns a numpy array of shape (1,).
    """
    qc = build_fcl_circuit()
    param_bindings = [{qc.parameters[0]: val} for val in theta_vals]
    simulator = Aer.get_backend("aer_simulator")
    job = execute(qc, simulator, shots=shots, parameter_binds=param_bindings)
    result = job.result()
    counts = result.get_counts(qc)
    total = sum(counts.values())
    expectation = sum(int(bit) * count / total for bit, count in counts.items())
    return np.array([expectation])


# --------------------------------------------------------------------------- #
# Combined quantum fraud detection interface
# --------------------------------------------------------------------------- #

def fraud_detection_quantum(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
    input_vals: Sequence[float],
    weight_vals: Sequence[float],
    shots: int = 1024
) -> np.ndarray:
    """
    High‑level helper that builds the fraud‑detection circuit, executes it
    with a SamplerQNN and returns the probability distribution.
    The input_params and layers are used only to bound parameters; the actual
    quantum circuit is constructed from input_vals and weight_vals.
    """
    # Build photonic‑style circuit (for bounds only)
    _ = build_fraud_detection_circuit(input_params, layers)
    # Execute SamplerQNN
    probs = sampler_qnn(input_vals, weight_vals, shots=shots)
    return probs


__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_circuit",
    "sampler_qnn",
    "build_fcl_circuit",
    "expectation_from_fcl",
    "fraud_detection_quantum"
]
