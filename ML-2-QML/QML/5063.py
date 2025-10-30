from __future__ import annotations

from typing import Iterable, List

import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import ZFeatureMap


class FraudDetectionModel:
    """
    Quantum counterpart of :class:`FraudDetectionModel`.  The circuit consists of:

        * a Z‑feature map that embeds the classical input
        * a variational ansatz with depth‑controlled layers
        * optional photonic‑style operations encoded as controlled‑Z and RY gates
        * measurement of a single qubit to obtain an expectation value

    The API mirrors the classical model: ``run(inputs, params)`` returns a scalar
    expectation value that can be used for classification.  The class can be
    extended to support different backends or measurement observables.
    """
    def __init__(self, num_qubits: int = 8, depth: int = 3, shots: int = 1024) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.shots = shots
        self.backend = Aer.get_backend("qasm_simulator")
        self.circuit, self.encoding, self.weights, self.observable = self._build_circuit()

    def _build_circuit(self) -> tuple[QuantumCircuit, List, List, SparsePauliOp]:
        # Encoding circuit: Z‑feature map
        feature_map = ZFeatureMap(self.num_qubits, reps=1, entanglement="full")
        # Parameterized variational ansatz
        encoding = ParameterVector("x", self.num_qubits)
        weights = ParameterVector("theta", self.num_qubits * self.depth)

        circuit = QuantumCircuit(self.num_qubits)
        # Encode data
        circuit.compose(feature_map, inplace=True)
        # Variational layers
        idx = 0
        for _ in range(self.depth):
            # Arbitrary entangling layer: RY rotations followed by CZ gates
            for q in range(self.num_qubits):
                circuit.ry(weights[idx], q)
                idx += 1
            for q in range(self.num_qubits - 1):
                circuit.cz(q, q + 1)
        # Observable: Z on first qubit
        observable = SparsePauliOp.from_list([("Z" + "I" * (self.num_qubits - 1), 1)])
        return circuit, list(encoding), list(weights), observable

    def run(self, inputs: Iterable[float], params: Iterable[float]) -> float:
        """
        Execute the circuit with the given input data and variational parameters.

        Parameters
        ----------
        inputs : Iterable[float]
            Classical feature vector to be embedded by the feature map.
        params : Iterable[float]
            Parameters for the variational ansatz.

        Returns
        -------
        float
            Expectation value of the observable (Z on qubit 0).
        """
        # Bind parameters
        bind_dict = {p: val for p, val in zip(self.encoding + self.weights, list(inputs) + list(params))}
        job = execute(
            self.circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[bind_dict],
        )
        result = job.result()
        counts = result.get_counts(self.circuit)
        # Map measurement outcomes to ±1 eigenvalues of Z
        exp = 0.0
        for outcome, count in counts.items():
            bit = int(outcome[-1])  # qubit 0 is the least significant bit
            eig = 1 if bit == 0 else -1
            exp += eig * count
        exp /= self.shots
        return float(exp)

__all__ = ["FraudDetectionModel"]
