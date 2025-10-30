"""Quantum QCNN implementation with feature‑map ansatz and pooling layers.

The architecture mirrors the classical structure but uses a parameterized
quantum circuit constructed from convolution and pooling sub‑circuits.
It also includes a fully connected quantum layer as a measurement on the
final qubit, inspired by the FCL quantum example.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator as StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_aer import AerSimulator

class QCNNGen351:
    """
    Quantum QCNN with convolution + pooling layers.

    Attributes
    ----------
    circuit : QuantumCircuit
        The full feature‑map + ansatz circuit.
    qnn : EstimatorQNN
        Variational QNN ready for training or inference.
    """

    def __init__(
        self,
        num_qubits: int = 8,
        backend: str | None = None,
        shots: int = 1024,
        seed: int | None = 12345,
    ) -> None:
        self.num_qubits = num_qubits
        self.shots = shots
        self.seed = seed
        self.backend = backend or AerSimulator(method="statevector")

        # Feature map
        self.feature_map = ZFeatureMap(num_qubits)

        # Ansätze
        self.ansatz = self._build_ansatz()

        # Combined circuit
        self.circuit = QuantumCircuit(num_qubits)
        self.circuit.compose(self.feature_map, range(num_qubits), inplace=True)
        self.circuit.compose(self.ansatz, range(num_qubits), inplace=True)

        # Observable (single‑qubit Z on last qubit)
        self.observable = SparsePauliOp.from_list([("Z" + "I" * (num_qubits - 1), 1)])

        # QNN
        self.qnn = EstimatorQNN(
            circuit=self.circuit.decompose(),
            observables=self.observable,
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            estimator=StatevectorEstimator(self.backend),
        )

    def _conv_circuit(self, params: ParameterVector) -> QuantumCircuit:
        """Single 2‑qubit convolution sub‑circuit."""
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        qc.cx(1, 0)
        qc.rz(np.pi / 2, 0)
        return qc

    def _pool_circuit(self, params: ParameterVector) -> QuantumCircuit:
        """Single 2‑qubit pooling sub‑circuit."""
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def _conv_layer(self, num_qubits: int, prefix: str) -> QuantumCircuit:
        """Build a convolution layer over all qubit pairs."""
        qc = QuantumCircuit(num_qubits)
        qubits = list(range(num_qubits))
        params = ParameterVector(prefix, length=num_qubits // 2 * 3)
        for i, (q1, q2) in enumerate(zip(qubits[0::2], qubits[1::2])):
            sub = self._conv_circuit(params[i * 3 : i * 3 + 3])
            qc.append(sub, [q1, q2])
            qc.barrier()
        return qc

    def _pool_layer(self, sources: list[int], sinks: list[int], prefix: str) -> QuantumCircuit:
        """Build a pooling layer mapping sources to sinks."""
        num = len(sources) + len(sinks)
        qc = QuantumCircuit(num)
        params = ParameterVector(prefix, length=len(sources) * 3)
        for i, (src, sink) in enumerate(zip(sources, sinks)):
            sub = self._pool_circuit(params[i * 3 : i * 3 + 3])
            qc.append(sub, [src, sink])
            qc.barrier()
        return qc

    def _build_ansatz(self) -> QuantumCircuit:
        """Assemble the full ansatz with three conv‑pool stages."""
        ansatz = QuantumCircuit(self.num_qubits)

        # First stage
        ansatz.append(self._conv_layer(self.num_qubits, "c1"), range(self.num_qubits))
        ansatz.append(self._pool_layer(list(range(self.num_qubits // 2)),
                                       list(range(self.num_qubits // 2, self.num_qubits)),
                                       "p1"),
                      range(self.num_qubits))

        # Second stage
        ansatz.append(self._conv_layer(self.num_qubits // 2, "c2"),
                      list(range(self.num_qubits // 2, self.num_qubits)))
        ansatz.append(self._pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p2"),
                      range(self.num_qubits // 2, self.num_qubits))

        # Third stage
        ansatz.append(self._conv_layer(self.num_qubits // 4, "c3"),
                      list(range(self.num_qubits // 2, self.num_qubits)))
        ansatz.append(self._pool_layer([0], [1], "p3"),
                      range(self.num_qubits // 2, self.num_qubits))

        return ansatz

    def run(self, inputs: np.ndarray) -> np.ndarray:
        """
        Execute the QCNN on a batch of inputs.

        Parameters
        ----------
        inputs : np.ndarray
            Array of shape (n_samples, num_qubits).

        Returns
        -------
        np.ndarray
            Array of expectation values (probabilities) for each sample.
        """
        return self.qnn.predict(inputs)

    def get_circuit(self) -> QuantumCircuit:
        """Return the full decomposed circuit for inspection."""
        return self.circuit.decompose()

__all__ = ["QCNNGen351"]
