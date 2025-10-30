"""Hybrid quantum FCL implementation inspired by QCNN and the original FCL example."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import Estimator
from typing import Iterable, Sequence


class FCL:
    """
    Quantum analogue of the classical FCL that uses a convolution‑style ansatz
    built from two‑qubit blocks, a feature map, and a single‑qubit observable.
    The interface matches the classical version: a ``run`` method that
    accepts a list of parameters and returns an expectation value.
    """

    def __init__(self, n_qubits: int = 4) -> None:
        self.n_qubits = n_qubits
        self.backend = Aer.get_backend("qasm_simulator")
        self.shots = 1024

        # Build the full circuit: feature map + convolution + pooling layers
        self.circuit = self._build_circuit()

        # Observable: single‑qubit Z on the first qubit
        self.observables = SparsePauliOp.from_list([("Z" + "I" * (n_qubits - 1), 1)])
        self.estimator = Estimator()
        self.qnn = EstimatorQNN(
            circuit=self.circuit,
            observables=self.observables,
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            estimator=self.estimator,
        )

    # ------------------------------------------------------------------
    # Circuit construction helpers
    # ------------------------------------------------------------------
    def _conv_block(self, params: ParameterVector, qubits: Sequence[int]) -> QuantumCircuit:
        """
        Two‑qubit convolution block used throughout the ansatz.
        """
        qc = QuantumCircuit(len(qubits))
        qc.rz(-np.pi / 2, qubits[1])
        qc.cx(qubits[1], qubits[0])
        qc.rz(params[0], qubits[0])
        qc.ry(params[1], qubits[1])
        qc.cx(qubits[0], qubits[1])
        qc.ry(params[2], qubits[1])
        qc.cx(qubits[1], qubits[0])
        qc.rz(np.pi / 2, qubits[0])
        return qc

    def _pool_block(self, params: ParameterVector, qubits: Sequence[int]) -> QuantumCircuit:
        """
        Two‑qubit pooling block that reduces entanglement.
        """
        qc = QuantumCircuit(len(qubits))
        qc.rz(-np.pi / 2, qubits[1])
        qc.cx(qubits[1], qubits[0])
        qc.rz(params[0], qubits[0])
        qc.ry(params[1], qubits[1])
        qc.cx(qubits[0], qubits[1])
        qc.ry(params[2], qubits[1])
        return qc

    def _build_circuit(self) -> QuantumCircuit:
        """
        Assemble the feature map and ansatz, mirroring the QCNN construction.
        """
        # Feature map
        self.feature_map = ZFeatureMap(self.n_qubits, reps=1, entanglement="full")
        circuit = QuantumCircuit(self.n_qubits)
        circuit.compose(self.feature_map, inplace=True)

        # Ansatz: three convolution‑pooling stages
        self.ansatz = QuantumCircuit(self.n_qubits)
        # Stage 1: 2‑qubit conv on (0,1) and (2,3)
        params1 = ParameterVector("c1", length=self.n_qubits // 2 * 3)
        for i in range(0, self.n_qubits, 2):
            block = self._conv_block(params1[i // 2 * 3 : i // 2 * 3 + 3], [i, i + 1])
            self.ansatz.append(block.to_instruction(), [i, i + 1])
        # Stage 2: pooling on (0,2) and (1,3)
        params2 = ParameterVector("p1", length=self.n_qubits // 2 * 3)
        for i in range(0, self.n_qubits // 2):
            block = self._pool_block(params2[i * 3 : i * 3 + 3], [i, i + self.n_qubits // 2])
            self.ansatz.append(block.to_instruction(), [i, i + self.n_qubits // 2])

        # Stage 3: conv on (0,1) again
        params3 = ParameterVector("c2", length=3)
        block = self._conv_block(params3, [0, 1])
        self.ansatz.append(block.to_instruction(), [0, 1])

        # Combine
        circuit.compose(self.ansatz, inplace=True)
        return circuit

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Execute the quantum circuit with the given parameters.

        Parameters
        ----------
        thetas : Iterable[float]
            Flattened list of parameters for all weight parameters in the ansatz.
            The length must match ``len(self.ansatz.parameters)``.

        Returns
        -------
        np.ndarray
            1‑D array containing the expectation value of the observable.
        """
        if len(thetas)!= len(self.ansatz.parameters):
            raise ValueError(
                f"Expected {len(self.ansatz.parameters)} parameters, got {len(thetas)}."
            )
        param_dict = dict(zip(self.ansatz.parameters, thetas))
        result = self.qnn.eval(param_dict)
        return np.array(result)

def FCL() -> FCL:
    """Factory that returns a pre‑configured quantum FCL instance."""
    return FCL()

__all__ = ["FCL", "FCL"]
