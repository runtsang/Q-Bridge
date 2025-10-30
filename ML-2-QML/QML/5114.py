"""Advanced quantum estimator that merges a quanvolution filter and a self‑attention circuit."""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.primitives import Estimator as StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from typing import Iterable, List, Sequence

# --------------------------------------------------------------------------- #
# 1. Combined quanvolution + self‑attention circuit
# --------------------------------------------------------------------------- #
class QuantumHybridCircuit:
    """
    Builds a circuit that first encodes a 2×2 image patch into a
    4‑qubit quanvolution block, then applies a 4‑qubit self‑attention style
    entanglement, and finally measures all qubits.
    """
    def __init__(self, conv_kernel: int = 2, attn_qubits: int = 4, shots: int = 1024) -> None:
        self.conv_n_qubits = conv_kernel ** 2
        self.attn_qubits = attn_qubits
        self.total_qubits = self.conv_n_qubits + self.attn_qubits
        self.shots = shots

        # Build the circuit
        self.qr = QuantumRegister(self.total_qubits, "q")
        self.cr = ClassicalRegister(self.total_qubits, "c")
        self.circuit = QuantumCircuit(self.qr, self.cr)

        # 1) Quanvolution block – angle‑encoding + random layers
        self.conv_params = [Parameter(f"conv_{i}") for i in range(self.conv_n_qubits)]
        for i in range(self.conv_n_qubits):
            self.circuit.ry(self.conv_params[i], i)
        self.circuit += random_circuit(self.conv_n_qubits, 2)

        # 2) Self‑attention style entanglement on the remaining qubits
        self.attn_params_rot = [Parameter(f"rot_{i}") for i in range(self.attn_qubits * 3)]
        self.attn_params_ent = [Parameter(f"ent_{i}") for i in range(self.attn_qubits - 1)]

        for i in range(self.attn_qubits):
            idx = self.conv_n_qubits + i
            self.circuit.rx(self.attn_params_rot[3 * i], idx)
            self.circuit.ry(self.attn_params_rot[3 * i + 1], idx)
            self.circuit.rz(self.attn_params_rot[3 * i + 2], idx)

        for i in range(self.attn_qubits - 1):
            self.circuit.crx(self.attn_params_ent[i],
                              self.conv_n_qubits + i,
                              self.conv_n_qubits + i + 1)

        # 3) Measurement of all qubits
        self.circuit.measure_all()

        # Backend and estimator
        self.backend = qiskit.Aer.get_backend("qasm_simulator")
        self.estimator = StatevectorEstimator()

        # Observables – average Z over all qubits
        self.observables = [SparsePauliOp.from_list([("Z" * self.total_qubits, 1)])]

        # Wrap into qiskit’s EstimatorQNN
        self._qnn = EstimatorQNN(
            circuit=self.circuit,
            observables=self.observables,
            input_params=self.conv_params,
            weight_params=self.attn_params_rot + self.attn_params_ent,
            estimator=self.estimator,
        )

    def evaluate(
        self,
        data: Sequence[float],
        weights: Sequence[float],
    ) -> List[complex]:
        """
        Evaluate the circuit for a single set of parameters.

        Parameters
        ----------
        data : Sequence[float]
            Length `conv_n_qubits`. Values are angle‑encoded into the
            quanvolution block.
        weights : Sequence[float]
            Length `attn_qubits * 3 + attn_qubits - 1`. These are the rotation
            and entanglement parameters of the self‑attention part.
        """
        if len(data)!= self.conv_n_qubits:
            raise ValueError("Data length must match quanvolution qubits.")
        if len(weights)!= len(self.attn_params_rot) + len(self.attn_params_ent):
            raise ValueError("Weight length mismatch for self‑attention parameters.")

        param_values = list(data) + list(weights)
        return self._qnn.evaluate(param_values)

# --------------------------------------------------------------------------- #
# 2. FastBaseEstimator wrapper (adapted from FastBaseEstimator.py)
# --------------------------------------------------------------------------- #
class FastBaseEstimator:
    """
    Lightweight evaluator for the combined quantum circuit.
    """
    def __init__(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

# --------------------------------------------------------------------------- #
# 3. The advanced quantum estimator
# --------------------------------------------------------------------------- #
class AdvancedEstimatorQNN:
    """
    Public quantum interface mirroring the classical class.
    It internally builds a hybrid circuit and exposes an `evaluate`
    method compatible with the classical `FastEstimator`.
    """
    def __init__(
        self,
        conv_kernel: int = 2,
        attn_qubits: int = 4,
        shots: int = 1024,
    ) -> None:
        self.circuit_builder = QuantumHybridCircuit(conv_kernel, attn_qubits, shots)
        self.estimator = FastBaseEstimator(self.circuit_builder.circuit)

    def evaluate(
        self,
        data: Sequence[float],
        weights: Sequence[float],
    ) -> List[List[complex]]:
        """
        Evaluate the circuit for a batch of data/weight pairs.

        Parameters
        ----------
        data : Sequence[float]
            Iterable of data vectors (each of length `conv_n_qubits`).
        weights : Sequence[float]
            Iterable of weight vectors (each of length
            `attn_qubits * 3 + attn_qubits - 1`).
        """
        # Flatten the parameter sets for FastBaseEstimator
        param_sets: List[Sequence[float]] = []
        for d, w in zip(data, weights):
            param_sets.append(list(d) + list(w))
        return self.estimator.evaluate(self.circuit_builder.observables, param_sets)

__all__ = ["AdvancedEstimatorQNN"]
