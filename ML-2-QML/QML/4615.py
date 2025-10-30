"""Quantum self‑attention module based on Qiskit, mirroring the classical
interface but using a parameterised circuit to compute attention‑style
expectations.  The class is designed to be interchangeable with
ClassicalSelfAttention in toy experiments.
"""

from __future__ import annotations

import dataclasses
import numpy as np
import qiskit
from qiskit import QuantumCircuit, assemble, transpile
from qiskit.providers.aer import AerSimulator
from typing import Iterable, Tuple


@dataclasses.dataclass
class FraudLayerParameters:
    """Quantum‑compatible fraud‑style parameters.
    Parameters are grouped per head; the class expects a list of floats
    with length 3 * num_heads (rx, ry, rz for each head) for the rotation
    part and a list of floats for the entanglement angles.
    """
    rotation: Tuple[float,...]
    entangle: Tuple[float,...]
    scale: float = 1.0
    shift: float = 0.0
    clip: float = 5.0


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


class QuantumSelfAttention:
    """Quantum self‑attention using a parametric Qiskit circuit.
    The circuit consists of a layer of single‑qubit rotations followed
    by a chain of controlled‑rotation gates that couple the qubits,
    mimicking the key/value entanglement step.  The expectation value
    of the Pauli‑Z operator on each qubit is interpreted as an
    attention weight.
    """
    def __init__(
        self,
        n_qubits: int,
        backend: qiskit.providers.BaseBackend | None = None,
        shots: int = 1024,
        fraud_params: FraudLayerParameters | None = None,
    ) -> None:
        self.n_qubits = n_qubits
        self.backend = backend or AerSimulator()
        self.shots = shots
        self.fraud_params = fraud_params or FraudLayerParameters(
            rotation=tuple(0.0 for _ in range(3 * n_qubits)),
            entangle=tuple(0.0 for _ in range(n_qubits - 1)),
        )

    def _build_circuit(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
    ) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        # single‑qubit rotations (rx, ry, rz) for each qubit
        for i in range(self.n_qubits):
            qc.rx(rotation_params[3 * i], i)
            qc.ry(rotation_params[3 * i + 1], i)
            qc.rz(rotation_params[3 * i + 2], i)
        # entanglement chain
        for i in range(self.n_qubits - 1):
            qc.crx(entangle_params[i], i, i + 1)
        qc.measure_all()
        return qc

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int | None = None,
    ) -> np.ndarray:
        """Execute the circuit for the supplied parameters and return
        an array of expectation values for each qubit.
        """
        if shots is None:
            shots = self.shots
        qc = self._build_circuit(rotation_params, entangle_params)
        compiled = transpile(qc, self.backend)
        qobj = assemble(compiled, shots=shots)
        res = self.backend.run(qobj).result()
        counts = res.get_counts()
        # compute Z expectation for each qubit
        exp_vals = []
        for i in range(self.n_qubits):
            probs = 0.0
            for bitstring, cnt in counts.items():
                bit = int(bitstring[::-1][i])  # little‑endian
                probs += ((-1) ** bit) * cnt
            probs /= shots
            exp_vals.append(probs)
        exp_arr = np.array(exp_vals)
        # apply fraud‑style scaling and clipping
        exp_arr = exp_arr * self.fraud_params.scale + self.fraud_params.shift
        exp_arr = np.clip(exp_arr, -self.fraud_params.clip, self.fraud_params.clip)
        return exp_arr


def SelfAttention() -> QuantumSelfAttention:
    """Convenience constructor used by the anchor path."""
    return QuantumSelfAttention(n_qubits=4)
