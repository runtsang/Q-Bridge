"""Quantum convolutional neural network built with Qiskit.

The implementation generalises the original QCNN by
* using a parameterised feature map (ZFeatureMap),
* stacking multiple entangling ansatz layers (RealAmplitudes),
* adding a multi‑observable read‑out to increase expressive power,
* and wrapping the whole circuit in an EstimatorQNN.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZFeatureMap, RealAmplitudes
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

class QCNNModel:
    """Hybrid quantum‑classical QCNN based on a feature map and an entangling ansatz.

    Parameters
    ----------
    feature_dim : int, default 8
        Number of qubits / dimensionality of the input data.
    layers : int, default 3
        Number of entangling ansatz layers.
    entanglement : str, default 'full'
        Entanglement pattern for RealAmplitudes ('full', 'linear', etc.).
    """

    def __init__(self, feature_dim: int = 8, layers: int = 3, entanglement: str = "full") -> None:
        self.feature_map = ZFeatureMap(feature_dim)
        self.ansatz = self._build_ansatz(feature_dim, layers, entanglement)

        # Build the full circuit: feature map + ansatz
        self.circuit = QuantumCircuit(feature_dim)
        self.circuit.compose(self.feature_map, range(feature_dim), inplace=True)
        self.circuit.compose(self.ansatz, range(feature_dim), inplace=True)

        # Multi‑observable read‑out: Z on each qubit weighted uniformly
        observables = []
        for i in range(feature_dim):
            pauli_str = "I" * i + "Z" + "I" * (feature_dim - i - 1)
            observables.append(SparsePauliOp.from_list([(pauli_str, 1.0)]))
        self.observable = observables

        estimator = Estimator()
        self.qnn = EstimatorQNN(
            circuit=self.circuit.decompose(),
            observables=self.observable,
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            estimator=estimator,
        )

    def _build_ansatz(self, dim: int, layers: int, entanglement: str) -> QuantumCircuit:
        """Construct a stack of RealAmplitudes layers."""
        qc = QuantumCircuit(dim)
        for _ in range(layers):
            layer = RealAmplitudes(dim, entanglement=entanglement, reps=1).to_instruction()
            qc.append(layer, range(dim))
        return qc

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """Evaluate the QCNN on a batch of inputs.

        Parameters
        ----------
        data : array‑like, shape (n_samples, feature_dim)
            Classical input data to be embedded by the feature map.

        Returns
        -------
        np.ndarray
            Expectation values for each sample.
        """
        return self.qnn(data)

def QCNN() -> QCNNModel:
    """Convenience factory mirroring the original seed."""
    return QCNNModel()

__all__ = ["QCNNModel", "QCNN"]
