"""Quantum fraud‑detection circuit that mirrors the photonic stack.

The module implements a qubit circuit that:
  1. Applies a quantum convolution sub‑circuit (derived from Conv.py).
  2. Encodes the photonic parameters into parameterised rotations.
  3. Adds a randomised entangling layer for expressive depth.
  4. Measures all qubits to obtain a single expectation value.

The design is deliberately modular so that the quantum filter can be
replaced or tuned independently from the main fraud‑detection block.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit.circuit import Parameter
from qiskit.circuit.random import random_circuit
from qiskit import QuantumCircuit, execute, Aer

# ----------------------------------------------------------------------
# 1. Parameter container (identical to the classical side)
# ----------------------------------------------------------------------
from dataclasses import dataclass
from typing import Iterable, Sequence

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

# ----------------------------------------------------------------------
# 2. Utility clipping
# ----------------------------------------------------------------------
def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

# ----------------------------------------------------------------------
# 3. Quantum convolution filter – identical to Conv.py
# ----------------------------------------------------------------------
def Conv() -> QuantumCircuit:
    """Return a sub‑circuit that implements a quantum convolution filter."""
    class QuanvCircuit:
        def __init__(self, kernel_size: int, threshold: float):
            self.n_qubits = kernel_size ** 2
            self.circuit = QuantumCircuit(self.n_qubits)
            # Parameterised rotation for each qubit
            self.theta = [Parameter(f"theta{i}") for i in range(self.n_qubits)]
            for i in range(self.n_qubits):
                self.circuit.rx(self.theta[i], i)
            self.circuit.barrier()
            self.circuit += random_circuit(self.n_qubits, 2)
            self.circuit.measure_all()

            self.backend = Aer.get_backend("qasm_simulator")
            self.shots = 100
            self.threshold = threshold

        def run(self, data: np.ndarray) -> float:
            """Encode a 2‑D array into rotation angles and obtain a probability."""
            data = np.reshape(data, (1, self.n_qubits))
            param_binds = []
            for dat in data:
                bind = {self.theta[i]: np.pi if val > self.threshold else 0 for i, val in enumerate(dat)}
                param_binds.append(bind)

            job = execute(
                self.circuit,
                self.backend,
                shots=self.shots,
                parameter_binds=param_binds,
            )
            result = job.result().get_counts(self.circuit)

            counts = 0
            for key, val in result.items():
                ones = sum(int(bit) for bit in key)
                counts += ones * val

            return counts / (self.shots * self.n_qubits)

    # Instantiate with a 2×2 kernel and a mid‑range threshold
    return QuanvCircuit(filter_size:=2, threshold=127)

# ----------------------------------------------------------------------
# 4. Layer encoding – maps photonic parameters to rotations
# ----------------------------------------------------------------------
def _apply_layer(
    circ: QuantumCircuit,
    params: FraudLayerParameters,
    clip: bool,
) -> None:
    # Rotation angles derived from photonic parameters
    theta = params.bs_theta
    phi = params.bs_phi
    if clip:
        theta = _clip(theta, 5.0)
        phi = _clip(phi, 5.0)

    circ.rx(theta, 0)
    circ.rx(phi, 1)

    for i, phase in enumerate(params.phases):
        circ.rz(_clip(phase, 5.0) if clip else phase, i)

    for i, (r, ph) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        circ.rx(_clip(r, 5.0) if clip else r, i)
        circ.rz(_clip(ph, 5.0) if clip else ph, i)

    for i, (r, ph) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        circ.rx(_clip(r, 5.0) if clip else r, i)
        circ.rz(_clip(ph, 5.0) if clip else ph, i)

    for i, k in enumerate(params.kerr):
        circ.rz(_clip(k, 1.0) if clip else k, i)

    # Add an entangling layer for depth
    circ += random_circuit(2, 2)

    circ.barrier()

# ----------------------------------------------------------------------
# 5. Full fraud‑detection circuit
# ----------------------------------------------------------------------
def build_fraud_detection_qiskit_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> QuantumCircuit:
    """Construct a qubit circuit that mirrors the photonic fraud‑detection stack."""
    n_qubits = 2
    circ = QuantumCircuit(n_qubits)

    # 5.1 Quantum convolution sub‑circuit
    conv_circ = Conv()
    circ.compose(conv_circ.circuit, inplace=True)

    # 5.2 Main stack – encode each layer
    _apply_layer(circ, input_params, clip=False)
    for layer in layers:
        _apply_layer(circ, layer, clip=True)

    circ.measure_all()
    return circ

__all__ = [
    "FraudLayerParameters",
    "Conv",
    "build_fraud_detection_qiskit_program",
]
