"""
Quantum hybrid fraud detection circuit that mirrors the classical
preprocessing and classification flow.  The circuit encodes the two
features with Ry gates, applies a stack of variational layers
parameterised by the `FraudLayerParameters`, and measures all qubits
to obtain class probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from qiskit import QuantumCircuit, Aer, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.providers import Backend
from qiskit.result import Result


@dataclass
class FraudLayerParameters:
    """
    Parameters describing a photonic‑inspired layer.  They are reused
    to initialise the variational parameters of the quantum circuit.
    """
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


class FraudDetectionHybrid:
    """
    Quantum implementation of the fraud detection model.  The circuit
    structure follows the classical preprocessing (Ry encoding),
    variational layers (Rx/Rz with entanglement) and a measurement
    that yields a probability distribution over the two classes.
    """
    def __init__(self, params: FraudLayerParameters, depth: int = 2, num_qubits: int = 2) -> None:
        self.params = params
        self.depth = depth
        self.num_qubits = num_qubits
        self.circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits, self.num_qubits)

        # Feature encoding: two Ry gates for the two input features
        qc.ry(self.params.bs_theta, 0)
        qc.ry(self.params.bs_phi, 1)

        # Variational layers
        for _ in range(self.depth):
            for q in range(self.num_qubits):
                qc.rz(self.params.displacement_phi[q], q)
                qc.rx(self.params.displacement_r[q], q)
            # Entanglement pattern (CZ between neighbours, closing the loop)
            for q in range(self.num_qubits - 1):
                qc.cz(q, q + 1)
            qc.cz(self.num_qubits - 1, 0)

        # Measurement of all qubits
        qc.measure(range(self.num_qubits), range(self.num_qubits))
        return qc

    def run(self, backend: Backend | None = None, shots: int = 1024) -> dict[str, int]:
        """
        Execute the circuit on a simulator or real backend.

        Parameters
        ----------
        backend : Backend, optional
            Qiskit backend; defaults to Aer qasm simulator.
        shots : int, default 1024
            Number of shots for the measurement.

        Returns
        -------
        dict[str, int]
            Raw measurement counts.
        """
        if backend is None:
            backend = Aer.get_backend("qasm_simulator")
        transpiled = transpile(self.circuit, backend)
        job = backend.run(transpiled, shots=shots)
        result: Result = job.result()
        return result.get_counts()

    def probabilities(self, counts: dict[str, int]) -> dict[str, float]:
        """
        Convert raw counts into class probabilities.  For a 2‑qubit
        circuit the two most probable bit‑strings are mapped to the
        two classes.
        """
        total = sum(counts.values())
        probs = {k: v / total for k, v in counts.items()}
        # Sort by probability and take the two highest
        sorted_items = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        return {f"class_{i}": p for i, (_, p) in enumerate(sorted_items[:2])}

__all__ = ["FraudLayerParameters", "FraudDetectionHybrid"]
