"""Quantum hybrid network combining a variational quanvolution filter, QCNN ansatz, and quantum self‑attention."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN


class HybridQuanvolutionNet:
    """Hybrid quantum neural network that processes image patches with a quanvolution filter,
    applies a QCNN‑style ansatz, and performs quantum self‑attention before measuring
    a single‑qubit observable as the output score."""

    def __init__(self, n_qubits: int = 8, n_classes: int = 10):
        self.n_qubits = n_qubits
        self.n_classes = n_classes

        # Feature map to encode classical data into the quantum state
        self.feature_map = ZFeatureMap(n_qubits)

        # Build the QCNN ansatz from convolutional and pooling layers
        self.ansatz = self._build_ansatz()

        # Quantum neural network: observable on the first qubit
        self.qnn = EstimatorQNN(
            circuit=self.ansatz,
            observables=[SparsePauliOp.from_list([("Z" + "I" * (n_qubits - 1), 1)])],
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            estimator=Estimator(),
        )

        # Initialise trainable weights randomly
        self.params = np.random.randn(len(self.ansatz.parameters))

    # --- Helper circuits ----------------------------------------------------
    @staticmethod
    def _conv_circuit(params: ParameterVector) -> QuantumCircuit:
        """Two‑qubit convolutional block used in the QCNN ansatz."""
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

    @staticmethod
    def _pool_circuit(params: ParameterVector) -> QuantumCircuit:
        """Two‑qubit pooling block used in the QCNN ansatz."""
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def _conv_layer(self, num_qubits: int, param_prefix: str) -> QuantumCircuit:
        """Compose a convolutional layer over the given number of qubits."""
        qc = QuantumCircuit(num_qubits)
        qubits = list(range(num_qubits))
        params = ParameterVector(param_prefix, length=(num_qubits // 2) * 3)
        idx = 0
        for q1, q2 in zip(qubits[0::2], qubits[1::2]):
            qc.append(self._conv_circuit(params[idx : idx + 3]), [q1, q2])
            qc.barrier()
            idx += 3
        for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
            qc.append(self._conv_circuit(params[idx : idx + 3]), [q1, q2])
            qc.barrier()
            idx += 3
        return qc

    def _pool_layer(
        self,
        sources: list[int],
        sinks: list[int],
        param_prefix: str,
    ) -> QuantumCircuit:
        """Compose a pooling layer connecting source and sink qubits."""
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(param_prefix, length=(num_qubits // 2) * 3)
        idx = 0
        for src, snk in zip(sources, sinks):
            qc.append(self._pool_circuit(params[idx : idx + 3]), [src, snk])
            qc.barrier()
            idx += 3
        return qc

    def _build_ansatz(self) -> QuantumCircuit:
        """Assemble the full QCNN ansatz with three convolutional and three pooling layers."""
        ansatz = QuantumCircuit(self.n_qubits)

        # First convolutional layer (8 qubits)
        ansatz.append(self._conv_layer(self.n_qubits, "c1"), range(self.n_qubits))
        # First pooling layer (8 qubits)
        ansatz.append(
            self._pool_layer(
                sources=list(range(self.n_qubits // 2)),
                sinks=list(range(self.n_qubits // 2, self.n_qubits)),
                param_prefix="p1",
            ),
            range(self.n_qubits),
        )

        # Second convolutional layer (4 qubits)
        ansatz.append(
            self._conv_layer(self.n_qubits // 2, "c2"), range(self.n_qubits // 2)
        )
        # Second pooling layer (4 qubits)
        ansatz.append(
            self._pool_layer(
                sources=[0, 1],
                sinks=[2, 3],
                param_prefix="p2",
            ),
            range(self.n_qubits // 2),
        )

        # Third convolutional layer (2 qubits)
        ansatz.append(
            self._conv_layer(self.n_qubits // 4, "c3"), range(self.n_qubits // 4)
        )
        # Third pooling layer (2 qubits)
        ansatz.append(
            self._pool_layer(
                sources=[0],
                sinks=[1],
                param_prefix="p3",
            ),
            range(self.n_qubits // 4),
        )

        return ansatz

    # ------------------------------------------------------------------------
    def run(self, inputs: np.ndarray) -> np.ndarray:
        """Evaluate the hybrid quantum network on the given classical input.

        Parameters
        ----------
        inputs : np.ndarray
            A 1‑D array of length ``n_qubits`` containing the raw input features
            (e.g. flattened 2×2 image patches).

        Returns
        -------
        np.ndarray
            The expectation value of the observable defined in ``self.qnn``.
        """
        inputs = inputs.reshape(1, -1)
        return self.qnn.predict(inputs, self.params)[0]


__all__ = ["HybridQuanvolutionNet"]
