"""Core quantum modules for the hybrid classifier.

This module provides:
- A qubit‑based variational circuit mirroring the classical encoder,
- A photonic fraud‑detection program built with Strawberry‑Fields,
- Shared metadata for joint optimisation.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate

from dataclasses import dataclass


@dataclass
class FraudLayerParameters:
    """Parameters describing a single photonic fraud layer."""
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]


def build_classifier_circuit(
    num_qubits: int,
    depth: int,
) -> Tuple[QuantumCircuit, Iterable[int], Iterable[int], List[SparsePauliOp]]:
    """Build a qubit‑based variational classifier.

    Parameters
    ----------
    num_qubits : int
        Number of qubits (should match the number of input features).
    depth : int
        Number of variational layers.

    Returns
    -------
    circuit : QuantumCircuit
        The constructed circuit.
    encoding : Iterable[int]
        Indices of the parameters that encode the classical features.
    weights : Iterable[int]
        Indices of the variational parameters.
    observables : List[SparsePauliOp]
        Pauli operators whose expectation values form the logits.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)

    # Data encoding
    for qubit, param in enumerate(encoding):
        circuit.rx(param, qubit)

    # Variational layers
    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        # Entangling pattern
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    # Observables for binary classification
    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]

    return circuit, list(encoding), list(weights), observables


def build_photonic_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> sf.Program:
    """Create a photonic fraud‑detection program mirroring the classical sub‑module.

    Parameters
    ----------
    input_params : FraudLayerParameters
        Parameters for the first photonic layer.
    layers : Iterable[FraudLayerParameters]
        Parameters for the subsequent layers.

    Returns
    -------
    program : sf.Program
        The constructed photonic program.
    """
    program = sf.Program(2)

    def _apply_layer(modes: List, params: FraudLayerParameters, clip: bool) -> None:
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

    with program.context as q:
        _apply_layer(q, input_params, clip=False)
        for layer in layers:
            _apply_layer(q, layer, clip=True)

    return program


def _clip(value: float, bound: float) -> float:
    """Clamp a scalar to the range [-bound, bound]."""
    return max(-bound, min(bound, value))
