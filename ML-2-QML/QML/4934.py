from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Sequence, List, Tuple
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector


class FraudDetectionHybrid:
    """Quantum fraud‑detection circuit mirroring the classical parameters."""

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

    def _apply_photonic_params(
        self,
        circuit: QuantumCircuit,
        params: "FraudDetectionHybrid.FraudLayerParameters",
        clip: bool,
    ) -> None:
        """Map photonic parameters to single‑qubit rotations in a qiskit circuit."""
        theta = params.bs_theta if not clip else np.clip(params.bs_theta, -5, 5)
        phi = params.bs_phi if not clip else np.clip(params.bs_phi, -5, 5)
        circuit.ry(theta, 0)
        circuit.rz(phi, 1)
        for i, phase in enumerate(params.phases):
            circuit.rz(phase, i)
        for i, (r, _) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
            val = r if not clip else np.clip(r, -5, 5)
            circuit.rx(val, i)
        for i, (r, _) in enumerate(zip(params.displacement_r, params.displacement_phi)):
            val = r if not clip else np.clip(r, -5, 5)
            circuit.rz(val, i)
        for i, k in enumerate(params.kerr):
            val = k if not clip else np.clip(k, -1, 1)
            circuit.rz(val, i)

    def build_quantum_fraud_program(
        self,
        input_params: "FraudDetectionHybrid.FraudLayerParameters",
        layers: Iterable["FraudDetectionHybrid.FraudLayerParameters"],
        num_qubits: int = 4,
        depth: int = 2,
    ) -> QuantumCircuit:
        """
        Build a variational qubit circuit that first encodes the fraud‑layer
        parameters and then applies a depth‑controlled ansatz.
        """
        circuit = QuantumCircuit(num_qubits)
        self._apply_photonic_params(circuit, input_params, clip=False)
        for p in layers:
            self._apply_photonic_params(circuit, p, clip=True)

        # Variational ansatz
        weights = ParameterVector("theta", num_qubits * depth)
        idx = 0
        for _ in range(depth):
            for q in range(num_qubits):
                circuit.ry(weights[idx], q)
                idx += 1
            for q in range(num_qubits - 1):
                circuit.cz(q, q + 1)

        observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)
        ]
        circuit.save_statevector()
        return circuit

    class FastBaseEstimator:
        """Evaluator that returns Pauli‑Z expectation values for each parameter set."""

        def __init__(self, circuit: QuantumCircuit) -> None:
            self._circuit = circuit
            self._params = list(circuit.parameters)

        def _bind(self, param_values: Sequence[float]) -> QuantumCircuit:
            if len(param_values)!= len(self._params):
                raise ValueError("Parameter count mismatch")
            mapping = dict(zip(self._params, param_values))
            return self._circuit.assign_parameters(mapping, inplace=False)

        def evaluate(
            self,
            observables: Iterable[SparsePauliOp],
            parameter_sets: Sequence[Sequence[float]],
        ) -> List[List[complex]]:
            results: List[List[complex]] = []
            for vals in parameter_sets:
                state = Statevector.from_instruction(self._bind(vals))
                row = [state.expectation_value(obs) for obs in observables]
                results.append(row)
            return results

    __all__ = ["FraudDetectionHybrid"]
