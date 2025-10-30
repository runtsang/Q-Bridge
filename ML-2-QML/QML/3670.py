"""Hybrid quantum fraud detection classifier.

Implements a Qiskit circuit that encodes data via RX gates, injects
photonic‑inspired parameters into a variational ansatz, and returns
Z‑observables for measurement.  The structure mirrors the classical
branch while leveraging Qiskit’s parameterized circuit API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


@dataclass
class FraudLayerParameters:
    """Parameters describing a single photonic layer, reused in the quantum branch."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


def _clip(value: float, bound: float) -> float:
    """Clamp a scalar to a symmetric interval."""
    return max(-bound, min(bound, value))


def _apply_fraud_params(circuit: QuantumCircuit,
                        params: FraudLayerParameters,
                        *,
                        clip: bool) -> None:
    """Map photonic parameters to standard qubit gates."""
    for qubit in range(circuit.num_qubits):
        theta = params.bs_theta if not clip else _clip(params.bs_theta, 5.0)
        phi = params.bs_phi if not clip else _clip(params.bs_phi, 5.0)
        circuit.ry(theta, qubit)
        circuit.rz(phi, qubit)

        # Phase rotation
        circuit.rz(params.phases[qubit], qubit)
        # Displacement encoded as RX
        circuit.rx(params.displacement_r[qubit], qubit)
        # Kerr non‑linearity simulated with RZ
        k = params.kerr[qubit]
        circuit.rz(k if not clip else _clip(k, 1.0), qubit)


def build_fraud_detection_circuit(
    num_qubits: int,
    depth: int,
    fraud_params: Iterable[FraudLayerParameters],
) -> tuple[QuantumCircuit, List[ParameterVector], List[ParameterVector], List[SparsePauliOp]]:
    """Construct a parameterised Qiskit circuit for the hybrid fraud‑detection model."""
    encoding = ParameterVector("x", num_qubits)
    variational = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    for q, enc in enumerate(encoding):
        circuit.rx(enc, q)

    var_idx = 0
    for param in fraud_params:
        _apply_fraud_params(circuit, param, clip=False)

        # Variational entanglement block
        for q in range(num_qubits):
            circuit.ry(variational[var_idx], q)
            var_idx += 1
        for q in range(num_qubits - 1):
            circuit.cz(q, q + 1)

    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]

    return circuit, list(encoding), list(variational), observables


class FraudDetectionClassifier:
    """Quantum counterpart of the hybrid fraud‑detection classifier.

    The circuit encodes the input through RX rotations, injects the
    photonic‑inspired parameters as a set of deterministic gates, and
    applies a depth‑controlled variational layer.  The returned
    `weight_params` correspond to the variational Ry angles.
    """
    def __init__(self,
                 num_qubits: int,
                 depth: int,
                 fraud_params: Iterable[FraudLayerParameters],
                 ) -> None:
        self.circuit, self.encoding, self.weight_params, self.observables = \
            build_fraud_detection_circuit(num_qubits, depth, fraud_params)

    @property
    def encoding_indices(self) -> List[int]:
        """Return indices of the encoding parameters for API parity."""
        return list(range(len(self.encoding)))

    @property
    def weight_sizes(self) -> List[int]:
        return [len(p) for p in self.weight_params]

__all__ = ["FraudLayerParameters", "FraudDetectionClassifier"]
