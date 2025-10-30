"""Quantum QCNN with integrated quantum kernel.

The module implements the full variational QCNN circuit from the
original Qiskit seed, and adds a state‑vector based quantum kernel
layer that can be used for kernel‑based learning.  It relies on
Qiskit Aer for state‑vector simulation and the Qiskit Machine
Learning EstimatorQNN for training.
"""

from __future__ import annotations

import numpy as np
from typing import Sequence

import torch
from qiskit import QuantumCircuit, Aer
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN


class QCNNHybridQNN:
    """
    Quantum QCNN that mirrors the classical architecture but
    operates on a variational circuit.  It also exposes a
    quantum kernel function based on state‑vector overlap.
    """

    def __init__(self) -> None:
        # Core components
        self.estimator = StatevectorEstimator()
        self.feature_map = ZFeatureMap(8)

        # Build the QCNN ansatz
        self.ansatz = self._build_ansatz()

        # Observable for the classification output
        self.observable = SparsePauliOp.from_list([("Z" * 7 + "I", 1)])

        # EstimatorQNN for training
        self.qnn = EstimatorQNN(
            circuit=self.ansatz.decompose(),
            observables=self.observable,
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            estimator=self.estimator,
        )

    def _build_ansatz(self) -> QuantumCircuit:
        """Construct the hierarchical QCNN circuit."""
        # Define a single convolution block
        def conv_circuit(params):
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

        # Convolution layer over pairs of qubits
        def conv_layer(num_qubits, param_prefix):
            qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
            qubits = list(range(num_qubits))
            param_index = 0
            params = ParameterVector(param_prefix, length=num_qubits * 3)
            for q1, q2 in zip(qubits[0::2], qubits[1::2]):
                qc.compose(conv_circuit(params[param_index : param_index + 3]), [q1, q2], inplace=True)
                qc.barrier()
                param_index += 3
            for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
                qc.compose(conv_circuit(params[param_index : param_index + 3]), [q1, q2], inplace=True)
                qc.barrier()
                param_index += 3
            return qc

        # Pooling block
        def pool_circuit(params):
            qc = QuantumCircuit(2)
            qc.rz(-np.pi / 2, 1)
            qc.cx(1, 0)
            qc.rz(params[0], 0)
            qc.ry(params[1], 1)
            qc.cx(0, 1)
            qc.ry(params[2], 1)
            return qc

        def pool_layer(sources, sinks, param_prefix):
            num_qubits = len(sources) + len(sinks)
            qc = QuantumCircuit(num_qubits, name="Pooling Layer")
            param_index = 0
            params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
            for source, sink in zip(sources, sinks):
                qc.compose(pool_circuit(params[param_index : param_index + 3]), [source, sink], inplace=True)
                qc.barrier()
                param_index += 3
            return qc

        # Assemble the full ansatz
        ansatz = QuantumCircuit(8, name="Ansatz")

        # First convolution & pooling
        ansatz.compose(conv_layer(8, "c1"), list(range(8)), inplace=True)
        ansatz.compose(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), list(range(8)), inplace=True)

        # Second convolution & pooling
        ansatz.compose(conv_layer(4, "c2"), list(range(4, 8)), inplace=True)
        ansatz.compose(pool_layer([0, 1], [2, 3], "p2"), list(range(4, 8)), inplace=True)

        # Third convolution & pooling
        ansatz.compose(conv_layer(2, "c3"), list(range(6, 8)), inplace=True)
        ansatz.compose(pool_layer([0], [1], "p3"), list(range(6, 8)), inplace=True)

        return ansatz

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict using the EstimatorQNN."""
        return self.qnn(x)

    def _statevector(self, x: torch.Tensor) -> np.ndarray:
        """Return the statevector for a single input vector."""
        # Build a circuit with the feature map and ansatz
        qc = QuantumCircuit(8)
        qc.compose(self.feature_map, range(8), inplace=True)
        qc.compose(self.ansatz, range(8), inplace=True)

        # Bind parameters from the feature map
        param_dict = {p: x[i].item() for i, p in enumerate(self.feature_map.parameters)}
        qc.assign_parameters(param_dict, inplace=True)

        # Simulate
        backend = Aer.get_backend("statevector_simulator")
        result = backend.run(qc).result()
        return result.get_statevector()

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """
        Quantum kernel based on squared overlap of statevectors.

        Parameters
        ----------
        a, b : Sequence[torch.Tensor]
            Sequences of input vectors.

        Returns
        -------
        np.ndarray
            Gram matrix of shape (len(a), len(b)).
        """
        sv_a = [self._statevector(x) for x in a]
        sv_b = [self._statevector(y) for y in b]
        gram = np.zeros((len(sv_a), len(sv_b)), dtype=float)
        for i, va in enumerate(sv_a):
            for j, vb in enumerate(sv_b):
                gram[i, j] = np.abs(np.vdot(va, vb)) ** 2
        return gram

def QCNN() -> QCNNHybridQNN:
    """Factory returning the quantum QCNN‑hybrid."""
    return QCNNHybridQNN()

__all__ = ["QCNNHybridQNN", "QCNN"]
