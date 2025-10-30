"""Hybrid quantum model combining photonic fraud detection parameters with a qubit variational circuit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


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


def _apply_photonic_params(circuit: QuantumCircuit, params: FraudLayerParameters, clip: bool) -> None:
    # Map photonic parameters to qubit gates
    # bs_theta, bs_phi -> RX and RZ rotations on qubit 0
    circuit.rx(params.bs_theta, 0)
    circuit.rz(params.bs_phi, 0)

    # phases -> RZ on each qubit
    for i, phase in enumerate(params.phases):
        circuit.rz(phase, i)

    # squeeze_r, squeeze_phi -> RY rotations
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        circuit.ry(r if not clip else _clip(r, 5), i)

    # displacement_r, displacement_phi -> RZ rotations
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        circuit.rz(r if not clip else _clip(r, 5), i)

    # kerr -> U3 gate with small angles
    for i, k in enumerate(params.kerr):
        circuit.u3(k if not clip else _clip(k, 1), 0, 0, i)


def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, List[ParameterVector], List[ParameterVector], List[SparsePauliOp]]:
    """Build a hybrid quantum classifier: photonic‑inspired parameterization + variational layers."""
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)
    circuit = QuantumCircuit(num_qubits)

    # Encoding
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    # Photonic‑inspired initial layer
    photonic_params = FraudLayerParameters(
        bs_theta=0.5,
        bs_phi=0.3,
        phases=(0.1, -0.1),
        squeeze_r=(0.2, 0.2),
        squeeze_phi=(0.0, 0.0),
        displacement_r=(1.0, 1.0),
        displacement_phi=(0.0, 0.0),
        kerr=(0.0, 0.0),
    )
    _apply_photonic_params(circuit, photonic_params, clip=False)

    # Variational layers
    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    # Observables
    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return circuit, list(encoding), list(weights), observables


__all__ = ["FraudLayerParameters", "build_classifier_circuit"]
