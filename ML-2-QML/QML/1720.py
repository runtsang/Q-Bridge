"""
Quantum Convolutional Neural Network (QCNN) implemented as a Qiskit EstimatorQNN subclass.

Features:
* Parameterised depth that controls the number of conv‑pool blocks.
* Optional noise model for realistic device simulation.
* Convenience ``train`` method using COBYLA and a simple binary loss.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator as EstimatorBackend
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_aer import AerSimulator, noise as noise_model
from qiskit import transpile
from qiskit.utils import QuantumInstance
from typing import Optional, Sequence


class QCNN(EstimatorQNN):
    """Quantum Convolutional Neural Network wrapper.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the feature map and ansatz.
    depth : int
        Number of conv‑pool pairs per channel.
    noise : Optional[noise_model.NoiseModel]
        Noise model to inject into the simulator.  If None, a noiseless
        Aer simulator is used.
    """

    def __init__(
        self,
        num_qubits: int = 8,
        depth: int = 3,
        noise: Optional[noise_model.NoiseModel] = None,
    ) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.noise = noise

        # Feature map
        self.feature_map = ZFeatureMap(num_qubits, reps=1, entanglement="full")
        self.input_params = self.feature_map.parameters

        # Build ansatz
        self.ansatz = self._build_ansatz(num_qubits, depth)
        self.weight_params = self.ansatz.parameters

        # Observable
        observable = SparsePauliOp.from_list([("Z" + "I" * (num_qubits - 1), 1)])

        # Backend
        backend = AerSimulator(noise_model=noise) if noise else AerSimulator()
        quantum_instance = QuantumInstance(backend, shots=1024, seed_simulator=42, seed_transpiler=42)

        super().__init__(
            circuit=self.ansatz.decompose(),
            observables=observable,
            input_params=self.input_params,
            weight_params=self.weight_params,
            estimator=EstimatorBackend(quantum_instance),
        )

    def _conv_block(self, params: ParameterVector, qubits: Sequence[int]) -> QuantumCircuit:
        """Single 2‑qubit convolution block."""
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

    def _pool_block(self, params: ParameterVector, qubits: Sequence[int]) -> QuantumCircuit:
        """Single 2‑qubit pooling block."""
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def _build_ansatz(self, num_qubits: int, depth: int) -> QuantumCircuit:
        """Construct the full QCNN ansatz."""
        qc = QuantumCircuit(num_qubits)
        qubit_pairs = [(i, i + 1) for i in range(0, num_qubits, 2)]

        param_index = 0
        for d in range(depth):
            conv_params = ParameterVector(f"c{d}_", length=len(qubit_pairs) * 3)
            for (q, r), idx in zip(qubit_pairs, range(len(qubit_pairs))):
                block = self._conv_block(conv_params[idx * 3 : idx * 3 + 3], [q, r])
                qc.compose(block, qubits=[q, r], inplace=True)
                qc.barrier()

            pool_params = ParameterVector(f"p{d}_", length=len(qubit_pairs) * 3)
            for (q, r), idx in zip(qubit_pairs, range(len(qubit_pairs))):
                block = self._pool_block(pool_params[idx * 3 : idx * 3 + 3], [q, r])
                qc.compose(block, qubits=[q, r], inplace=True)
                qc.barrier()
        return qc

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 10,
        optimizer: str = "COBYLA",
    ) -> None:
        """Simple training loop using the specified optimizer."""
        # Wrap the EstimatorQNN into a classifier
        classifier = NeuralNetworkClassifier(
            estimator_qnn=self,
            optimizer=COBYLA() if optimizer == "COBYLA" else COBYLA(),
            loss="cross_entropy",
            training=True,
        )
        classifier.fit(X, y, epochs=epochs)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary labels for the given data."""
        probs = self.predict_proba(X)
        return (probs > threshold).astype(int)

__all__ = ["QCNN"]
