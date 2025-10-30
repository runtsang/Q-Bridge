"""Quantum hybrid QCNN integrating quantum convolutions and a quanvolution filter."""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.circuit.random import random_circuit

class QCNNHybrid:
    """
    Quantum QCNN that optionally prefixes a quanvolution filter (Conv)
    and then applies the standard QCNN layers.
    """
    def __init__(
        self,
        input_dim: int = 8,
        use_quanv: bool = False,
        quanv_kernel: int = 2,
        backend: str = "qasm_simulator",
        shots: int = 100,
        quanv_threshold: float = 127,
    ) -> None:
        qiskit.algorithms.utils.algorithm_globals.random_seed = 12345
        self.input_dim = input_dim
        self.use_quanv = use_quanv
        self.backend = qiskit.Aer.get_backend(backend)
        self.shots = shots
        self.quanv_threshold = quanv_threshold
        self.quanv_kernel = quanv_kernel

        # Build the base QCNN circuit
        self.feature_map = ZFeatureMap(input_dim)
        self.ansatz = self._build_ansatz()
        self.observable = SparsePauliOp.from_list([("Z" + "I" * (input_dim - 1), 1)])

        # Wrap with quanvolution filter if requested
        if self.use_quanv:
            self.quanv_circuit = self._build_quanv()
        else:
            self.quanv_circuit = None

        # Final QNN
        self.qnn = EstimatorQNN(
            circuit=self._compose_full_circuit(),
            observables=self.observable,
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            estimator=Estimator(
                backend=self.backend,
                shots=self.shots,
            ),
        )

    def _build_quanv(self) -> QuantumCircuit:
        """Construct a quantum filter identical to the Conv.py quanv circuit."""
        n_qubits = self.quanv_kernel ** 2
        qc = QuantumCircuit(n_qubits)
        thetas = [qiskit.circuit.Parameter(f"theta{i}") for i in range(n_qubits)]
        for i in range(n_qubits):
            qc.rx(thetas[i], i)
        qc.barrier()
        qc += random_circuit(n_qubits, 2)
        qc.measure_all()
        return qc

    def _build_ansatz(self) -> QuantumCircuit:
        """Construct the convolution / pooling ansatz of the QCNN."""
        ansatz = QuantumCircuit(self.input_dim, name="Ansatz")

        # First Convolutional Layer
        ansatz.compose(self._conv_layer(self.input_dim, "c1"), range(self.input_dim), inplace=True)

        # First Pooling Layer
        ansatz.compose(
            self._pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"),
            range(self.input_dim),
            inplace=True,
        )

        # Second Convolutional Layer
        ansatz.compose(
            self._conv_layer(self.input_dim // 2, "c2"),
            range(self.input_dim // 2, self.input_dim),
            inplace=True,
        )

        # Second Pooling Layer
        ansatz.compose(
            self._pool_layer([0, 1], [2, 3], "p2"),
            range(self.input_dim // 2, self.input_dim),
            inplace=True,
        )

        # Third Convolutional Layer
        ansatz.compose(
            self._conv_layer(self.input_dim // 4, "c3"),
            range(self.input_dim // 4 * 3, self.input_dim),
            inplace=True,
        )

        # Third Pooling Layer
        ansatz.compose(
            self._pool_layer([0], [1], "p3"),
            range(self.input_dim // 4 * 3, self.input_dim),
            inplace=True,
        )
        return ansatz

    def _conv_layer(self, num_qubits: int, param_prefix: str) -> QuantumCircuit:
        """Return a convolutional layer instruction."""
        qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
        params = ParameterVector(param_prefix, length=num_qubits * 3)
        idx = 0
        for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
            sub = self._conv_circuit(params[idx : idx + 3])
            qc.append(sub.to_instruction(), [q1, q2])
            qc.barrier()
            idx += 3
        return qc

    def _conv_circuit(self, params) -> QuantumCircuit:
        """Single 2‑qubit convolution sub‑circuit."""
        sub = QuantumCircuit(2)
        sub.rz(-np.pi / 2, 1)
        sub.cx(1, 0)
        sub.rz(params[0], 0)
        sub.ry(params[1], 1)
        sub.cx(0, 1)
        sub.ry(params[2], 1)
        sub.cx(1, 0)
        sub.rz(np.pi / 2, 0)
        return sub

    def _pool_layer(self, sources, sinks, param_prefix: str) -> QuantumCircuit:
        """Return a pooling layer instruction."""
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits, name="Pooling Layer")
        params = ParameterVector(param_prefix, length=(num_qubits // 2) * 3)
        idx = 0
        for src, sink in zip(sources, sinks):
            sub = self._pool_circuit(params[idx : idx + 3])
            qc.append(sub.to_instruction(), [src, sink])
            qc.barrier()
            idx += 3
        return qc

    def _pool_circuit(self, params) -> QuantumCircuit:
        """Single 2‑qubit pooling sub‑circuit."""
        sub = QuantumCircuit(2)
        sub.rz(-np.pi / 2, 1)
        sub.cx(1, 0)
        sub.rz(params[0], 0)
        sub.ry(params[1], 1)
        sub.cx(0, 1)
        sub.ry(params[2], 1)
        return sub

    def _compose_full_circuit(self) -> QuantumCircuit:
        """Combine feature map, optional quanv, and ansatz."""
        circuit = QuantumCircuit(self.input_dim)
        circuit.compose(self.feature_map, range(self.input_dim), inplace=True)
        if self.quanv_circuit:
            # Embed the quanv filter on the first few qubits
            circuit.compose(self.quanv_circuit, range(self.quanv_circuit.num_qubits), inplace=True)
        circuit.compose(self.ansatz, range(self.input_dim), inplace=True)
        return circuit

    def get_qnn(self) -> EstimatorQNN:
        """Return the constructed EstimatorQNN."""
        return self.qnn

__all__ = ["QCNNHybrid"]
