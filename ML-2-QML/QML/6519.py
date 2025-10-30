"""Quantum QCNN with embedded RBF‑style kernel.

This module implements a variational quantum circuit that
reproduces the convolution‑pooling structure of the
classical seed while providing a quantum kernel evaluation
via state‑vector overlap.  The `kernel_value` method
returns a similarity measure that directly mirrors the
classical RBF kernel used in :mod:`QCNN__gen353.ml_code`.
"""

import numpy as np
import torch
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.utils import algorithm_globals
from typing import List


class QCNNHybrid:
    """Quantum convolutional neural network with an RBF‑style kernel."""

    def __init__(self, n_qubits: int = 8, seed: int = 12345) -> None:
        algorithm_globals.random_seed = seed
        self.n_qubits = n_qubits

        self.feature_map = ZFeatureMap(n_qubits)
        self._build_ansatz()
        self._build_qnn()

    # ------------------------------------------------------------------
    #  Convolution / pooling primitives
    # ------------------------------------------------------------------
    @staticmethod
    def _conv_circuit(params: ParameterVector) -> QuantumCircuit:
        """Unitary used in each convolution block."""
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
        """Unitary used in each pooling block."""
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def _conv_layer(self, num_qubits: int, prefix: str) -> QuantumCircuit:
        """Wraps convolution blocks over a register of qubits."""
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(f"{prefix}", length=num_qubits * 3)
        for i in range(0, num_qubits, 2):
            sub = self._conv_circuit(params[i * 3 : (i + 2) * 3])
            qc.compose(sub, [i, i + 1], inplace=True)
            qc.barrier()
        return qc

    def _pool_layer(self, sources: List[int], sinks: List[int], prefix: str) -> QuantumCircuit:
        """Wraps pooling blocks over specified qubit pairs."""
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(f"{prefix}", length=len(sources) * 3)
        for idx, (s, t) in enumerate(zip(sources, sinks)):
            sub = self._pool_circuit(params[idx * 3 : (idx + 1) * 3])
            qc.compose(sub, [s, t], inplace=True)
            qc.barrier()
        return qc

    # ------------------------------------------------------------------
    #  Build the ansatz
    # ------------------------------------------------------------------
    def _build_ansatz(self) -> None:
        """Constructs the full variational ansatz with conv/pool layers."""
        ansatz = QuantumCircuit(self.n_qubits)
        # First conv and pool
        ansatz.compose(self._conv_layer(self.n_qubits, "c1"), inplace=True)
        ansatz.compose(
            self._pool_layer(list(range(0, 4)), list(range(4, 8)), "p1"), inplace=True
        )

        # Second conv and pool
        ansatz.compose(self._conv_layer(4, "c2"), inplace=True)
        ansatz.compose(self._pool_layer([0, 1], [2, 3], "p2"), inplace=True)

        # Third conv and pool
        ansatz.compose(self._conv_layer(2, "c3"), inplace=True)
        ansatz.compose(self._pool_layer([0], [1], "p3"), inplace=True)

        self.ansatz = ansatz

    # ------------------------------------------------------------------
    #  Build the EstimatorQNN
    # ------------------------------------------------------------------
    def _build_qnn(self) -> None:
        """Wraps the circuit in an EstimatorQNN for supervised tasks."""
        circuit = QuantumCircuit(self.n_qubits)
        circuit.compose(self.feature_map, inplace=True)
        circuit.compose(self.ansatz, inplace=True)

        observable = SparsePauliOp.from_list([("Z" + "I" * (self.n_qubits - 1), 1)])
        self.estimator = Estimator()

        self.qnn = EstimatorQNN(
            circuit=circuit.decompose(),
            observables=observable,
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            estimator=self.estimator,
        )

    # ------------------------------------------------------------------
    #  Forward pass
    # ------------------------------------------------------------------
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Compute the network output for a batch of inputs.

        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of shape (batch, n_qubits).

        Returns
        -------
        torch.Tensor
            Sigmoid‑activated predictions of shape (batch, 1).
        """
        return torch.sigmoid(self.qnn(inputs))

    # ------------------------------------------------------------------
    #  Quantum kernel evaluation
    # ------------------------------------------------------------------
    def kernel_value(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the quantum kernel value between two feature vectors.

        The kernel is defined as the squared overlap between the
        statevectors produced by the feature map for `x` and `y`.
        """
        # Encode x
        circ_x = self.feature_map.copy()
        circ_x.bind_parameters(
            {p: val for p, val in zip(self.feature_map.parameters, x.tolist())}
        )
        state_x = Statevector(circ_x)

        # Encode y
        circ_y = self.feature_map.copy()
        circ_y.bind_parameters(
            {p: val for p, val in zip(self.feature_map.parameters, y.tolist())}
        )
        state_y = Statevector(circ_y)

        # Overlap squared
        overlap = abs(state_x.inner(state_y)) ** 2
        return torch.tensor(overlap, dtype=torch.float)

    def kernel_matrix(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Compute the Gram matrix between two sets of feature vectors.

        Parameters
        ----------
        a : torch.Tensor
            Tensor of shape (n, n_qubits).
        b : torch.Tensor
            Tensor of shape (m, n_qubits).

        Returns
        -------
        torch.Tensor
            Gram matrix of shape (n, m).
        """
        n, m = a.shape[0], b.shape[0]
        K = torch.empty(n, m)
        for i in range(n):
            for j in range(m):
                K[i, j] = self.kernel_value(a[i], b[j])
        return K


def create_QCNNHybrid() -> QCNNHybrid:
    """Factory returning a default :class:`QCNNHybrid` instance."""
    return QCNNHybrid()


__all__ = ["QCNNHybrid", "create_QCNNHybrid"]
