"""Hybrid classifier quantum implementation.

The class exposes two complementary quantum back‑ends:
* a Qiskit ansatz that mirrors the classical feed‑forward depth.
* a Strawberry Fields photonic program that reproduces the fraud‑detection
  operations used in the classical model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

# Re‑use the photonic parameter dataclass
@dataclass
class FraudLayerParameters:
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

def _apply_photonic_layer(modes: Iterable, params: FraudLayerParameters, *, clip: bool) -> None:
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

class HybridClassifier:
    """
    Quantum counterpart to :class:`ml.HybridClassifier`.  It provides:

    * build_quantum_circuit – a parameterised Qiskit circuit with depth‑wise
      Ry rotations and CZ entanglement.
    * build_photonic_program – a Strawberry Fields program that reproduces the
      fraud‑detection photonic layer sequence.

    Parameters
    ----------
    num_qubits : int
        Number of qubits / photonic modes.
    depth : int
        Depth of the Qiskit ansatz.
    fraud_params : Iterable[FraudLayerParameters]
        Parameters for the photonic layer sequence.
    """

    def __init__(
        self,
        num_qubits: int,
        depth: int,
        fraud_params: Iterable[FraudLayerParameters],
    ) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.fraud_params = list(fraud_params)

    def build_quantum_circuit(self) -> Tuple[QuantumCircuit, List[ParameterVector], List[ParameterVector], List[SparsePauliOp]]:
        """Return a Qiskit circuit together with its encoding, variational parameters
        and measurement observables."""
        encoding = ParameterVector("x", self.num_qubits)
        weights = ParameterVector("theta", self.num_qubits * self.depth)

        circuit = QuantumCircuit(self.num_qubits)
        for param, qubit in zip(encoding, range(self.num_qubits)):
            circuit.rx(param, qubit)

        idx = 0
        for _ in range(self.depth):
            for qubit in range(self.num_qubits):
                circuit.ry(weights[idx], qubit)
                idx += 1
            for qubit in range(self.num_qubits - 1):
                circuit.cz(qubit, qubit + 1)

        observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (self.num_qubits - i - 1))
            for i in range(self.num_qubits)
        ]
        return circuit, list(encoding), list(weights), observables

    def build_photonic_program(self) -> sf.Program:
        """Return a Strawberry Fields program that implements the fraud‑detection
        photonic layers described by ``self.fraud_params``."""
        program = sf.Program(self.num_qubits)
        with program.context as q:
            _apply_photonic_layer(q, self.fraud_params[0], clip=False)
            for layer in self.fraud_params[1:]:
                _apply_photonic_layer(q, layer, clip=True)
        return program

__all__ = ["HybridClassifier", "FraudLayerParameters"]
