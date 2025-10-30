from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals

class HybridQCNN:
    """Quantum neural network mirroring the classical HybridQCNN.

    The circuit consists of an RX feature‑map followed by a stack of
    convolution–pooling blocks.  The number of layers is controlled by
    ``depth`` and each block has the same 3‑parameter gate pattern as in
    the original QCNN seed.  This design keeps the parameter count of
    the quantum model in line with the classical depth, enabling fair
    comparisons and hybrid training.
    """
    def __init__(self, num_qubits: int, depth: int = 3):
        algorithm_globals.random_seed = 12345
        self.num_qubits = num_qubits
        self.depth = depth
        self.estimator = Estimator()
        self.circuit = self._build_ansatz()
        self.qnn = EstimatorQNN(
            circuit=self.circuit.decompose(),
            observables=self._build_observables(),
            input_params=self._encoding_parameters(),
            weight_params=self._weight_parameters(),
            estimator=self.estimator,
        )

    def _encoding_parameters(self):
        return ParameterVector("x", self.num_qubits).parameters

    def _weight_parameters(self):
        return ParameterVector("theta", self.num_qubits * self.depth).parameters

    def _conv_block(self, params, qubits):
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

    def _pool_block(self, params, qubits):
        qc = QuantumCircuit(len(qubits))
        qc.rz(-np.pi / 2, qubits[1])
        qc.cx(qubits[1], qubits[0])
        qc.rz(params[0], qubits[0])
        qc.ry(params[1], qubits[1])
        qc.cx(qubits[0], qubits[1])
        qc.ry(params[2], qubits[1])
        return qc

    def _build_ansatz(self):
        fm = QuantumCircuit(self.num_qubits)
        for i in range(self.num_qubits):
            fm.rx(ParameterVector("x", self.num_qubits)[i], i)

        ansatz = QuantumCircuit(self.num_qubits)
        qubits = list(range(self.num_qubits))
        for d in range(self.depth):
            # Convolution on adjacent pairs
            for i in range(0, len(qubits) - 1, 2):
                block = self._conv_block(
                    ParameterVector(f"c{d}_{i}", 3).parameters,
                    qubits[i:i+2]
                )
                ansatz.compose(block, qubits[i:i+2], inplace=True)
            # Pooling on the same pairs
            for i in range(0, len(qubits) - 1, 2):
                block = self._pool_block(
                    ParameterVector(f"p{d}_{i}", 3).parameters,
                    qubits[i:i+2]
                )
                ansatz.compose(block, qubits[i:i+2], inplace=True)

        return fm + ansatz

    def _build_observables(self):
        return [SparsePauliOp.from_list([("Z" + "I" * (self.num_qubits - 1), 1)])]

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Return sigmoid‑transformed predictions for the quantum model."""
        return np.squeeze(self.qnn.predict(inputs), axis=1)

def HybridQCNNFactory(num_qubits: int, depth: int = 3) -> HybridQCNN:
    return HybridQCNN(num_qubits, depth)

__all__ = ["HybridQCNN", "HybridQCNNFactory"]
