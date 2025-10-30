"""FraudDetectionHybrid – quantum implementation.

The quantum counterpart builds a Qiskit circuit that
1. performs a quantum self‑attention block,
2. applies a variational classification ansatz, and
3. returns the full circuit for execution or further processing.

The interface mirrors the classical build_classical method, so
parameter tensors from the PyTorch model can be injected
directly if a hybrid training loop is desired.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


@dataclass
class FraudLayerParameters:
    """Placeholder – parameters are not used in the quantum model."""
    bs_theta: float = 0.0
    bs_phi: float = 0.0
    phases: tuple[float, float] = (0.0, 0.0)
    squeeze_r: tuple[float, float] = (0.0, 0.0)
    squeeze_phi: tuple[float, float] = (0.0, 0.0)
    displacement_r: tuple[float, float] = (0.0, 0.0)
    displacement_phi: tuple[float, float] = (0.0, 0.0)
    kerr: tuple[float, float] = (0.0, 0.0)


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


class QuantumSelfAttention:
    """Encapsulates a self‑attention style block implemented in Qiskit."""

    def __init__(self, n_qubits: int) -> None:
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _build_circuit(
        self, rotation_params: np.ndarray, entangle_params: np.ndarray
    ) -> QuantumCircuit:
        circuit = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        circuit.measure(self.qr, self.cr)
        return circuit

    def run(
        self,
        backend,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ) -> dict:
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = qiskit.execute(circuit, backend, shots=shots)
        return job.result().get_counts(circuit)


def build_classifier_circuit(num_qubits: int, depth: int) -> tuple[
    QuantumCircuit, Sequence[ParameterVector], Sequence[ParameterVector], list[SparsePauliOp]
]:
    """Construct the variational classification ansatz used in the quantum model."""
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]
    return circuit, list(encoding), list(weights), observables


class FraudDetectionHybrid:
    """
    Unified interface that returns a Qiskit circuit mirroring the
    classical fraud‑detection pipeline.  The `build_quantum` method
    attaches a quantum self‑attention block before the classifier ansatz.
    """

    def __init__(
        self,
        num_qubits: int = 4,
        depth: int = 2,
        fraud_params: Iterable[FraudLayerParameters] | None = None,
    ) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.fraud_params = list(fraud_params) if fraud_params else []

    def build_quantum(self) -> tuple[
        QuantumCircuit, Sequence[ParameterVector], Sequence[ParameterVector], list[SparsePauliOp]
    ]:
        """Return a full Qiskit circuit for fraud detection."""
        # Self‑attention block
        sa = QuantumSelfAttention(self.num_qubits)
        rotation_params = np.random.rand(3 * self.num_qubits)
        entangle_params = np.random.rand(self.num_qubits - 1)

        sa_circuit = sa._build_circuit(rotation_params, entangle_params)

        # Classification ansatz
        cls_circuit, enc_params, var_params, observables = build_classifier_circuit(
            self.num_qubits, self.depth
        )

        # Combine
        full_circuit = QuantumCircuit(self.num_qubits)
        full_circuit.compose(sa_circuit, inplace=True)
        full_circuit.compose(cls_circuit, inplace=True)

        return full_circuit, enc_params, var_params, observables


__all__ = ["FraudLayerParameters", "QuantumSelfAttention", "build_classifier_circuit",
           "FraudDetectionHybrid"]
