"""Quantum hybrid classifier built with Qiskit, optionally including fraud‑detection inspired photonic layers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


@dataclass
class FraudLayerParams:
    """Parameters for a fraud‑detection style photonic layer (used for consistency)."""
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


def _apply_photonic_layer(circuit: QuantumCircuit, params: FraudLayerParams, clip: bool = False) -> None:
    """Emulate a photonic layer with equivalent Qiskit rotations for demonstration."""
    theta, phi = params.bs_theta, params.bs_phi
    circuit.ry(theta, 0)
    circuit.rz(phi, 0)
    for i, (r, p) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        circuit.rx(_clip(r, 5.0), i)
        circuit.rz(_clip(p, 5.0), i)
    for i, (r, p) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        circuit.rx(_clip(r, 5.0), i)
        circuit.rz(_clip(p, 5.0), i)
    for i, k in enumerate(params.kerr):
        circuit.rz(_clip(k, 1.0), i)


def build_quantum_circuit(
    num_qubits: int,
    depth: int,
    fraud_layer: Optional[FraudLayerParams] = None,
) -> Tuple[QuantumCircuit, List[ParameterVector], List[ParameterVector], List[SparsePauliOp]]:
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    # Input encoding
    for idx, qubit in enumerate(range(num_qubits)):
        circuit.rx(encoding[idx], qubit)

    # Variational ansatz
    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    # Optional fraud‑detection inspired photonic layer emulation
    if fraud_layer is not None:
        _apply_photonic_layer(circuit, fraud_layer)

    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]
    return circuit, [encoding], [weights], observables


class HybridClassifier:
    """Quantum hybrid classifier that mirrors the classical interface."""
    def __init__(
        self,
        num_qubits: int,
        depth: int = 1,
        fraud_layer: Optional[FraudLayerParams] = None,
    ) -> None:
        self.circuit, self.encodings, self.weights, self.observables = build_quantum_circuit(
            num_qubits, depth, fraud_layer
        )

    def get_circuit(self) -> QuantumCircuit:
        return self.circuit

    def get_parameters(self) -> Tuple[List[ParameterVector], List[ParameterVector]]:
        return self.encodings, self.weights

    def get_observables(self) -> List[SparsePauliOp]:
        return self.observables


__all__ = ["HybridClassifier", "FraudLayerParams"]
