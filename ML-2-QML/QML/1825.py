"""Quantum QCNN with modular convolution and pooling layers.

The implementation builds a variational ansatz that mirrors the
classical QCNN structure while providing a flexible pooling
strategy.  It exposes a lightweight hybrid training interface
and can be augmented with a classical head for post‑processing.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit.utils import algorithm_globals


class QCNN:
    """Quantum Circuit Neural Network inspired by the classical QCNN.

    Parameters
    ----------
    num_qubits : int
        Number of qubits used in the ansatz (default 8).
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, num_qubits: int = 8, seed: int = 12345) -> None:
        self.num_qubits = num_qubits
        self.seed = seed
        algorithm_globals.random_seed = seed
        self.estimator = Estimator()
        self.feature_map = ZFeatureMap(num_qubits)
        self.circuit = self._build_ansatz()
        self.qnn = self._build_qnn()

    # ---------------------------------------------------------------------
    #  Convolution and pooling primitives
    # ---------------------------------------------------------------------
    def _conv_circuit(self, params: ParameterVector) -> QuantumCircuit:
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

    def _conv_layer(self, num_qubits: int, param_prefix: str) -> QuantumCircuit:
        """Builds a convolutional layer that pairs adjacent qubits."""
        qc = QuantumCircuit(num_qubits)
        qubits = list(range(num_qubits))
        params = ParameterVector(param_prefix, length=num_qubits * 3)
        for i in range(0, num_qubits, 2):
            qc.compose(
                self._conv_circuit(params[i * 3 : (i + 1) * 3]), [qubits[i], qubits[i + 1]],
                inplace=True,
            )
        return qc

    def _pool_circuit(self, params: ParameterVector) -> QuantumCircuit:
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def _pool_layer(
        self, sources: list[int], sinks: list[int], param_prefix: str
    ) -> QuantumCircuit:
        """Builds a pooling layer that reduces the qubit count."""
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(param_prefix, length=len(sources) * 3)
        for src, sink, p_start in zip(sources, sinks, range(0, len(sources) * 3, 3)):
            qc.compose(
                self._pool_circuit(params[p_start : p_start + 3]),
                [src, sink],
                inplace=True,
            )
        return qc

    # ---------------------------------------------------------------------
    #  Ansatz construction
    # ---------------------------------------------------------------------
    def _build_ansatz(self) -> QuantumCircuit:
        """Constructs the full variational ansatz with alternating
        convolution and pooling layers.
        """
        qc = QuantumCircuit(self.num_qubits)
        qc.compose(self.feature_map, range(self.num_qubits), inplace=True)

        # Convolution + pooling stages
        qc.compose(self._conv_layer(self.num_qubits, "c1"), range(self.num_qubits), inplace=True)
        qc.compose(self._pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), range(self.num_qubits), inplace=True)

        qc.compose(self._conv_layer(self.num_qubits // 2, "c2"), range(self.num_qubits // 2, self.num_qubits), inplace=True)
        qc.compose(self._pool_layer([0, 1], [2, 3], "p2"), range(self.num_qubits // 2, self.num_qubits), inplace=True)

        qc.compose(self._conv_layer(self.num_qubits // 4, "c3"), range(self.num_qubits // 4 * 3, self.num_qubits), inplace=True)
        qc.compose(self._pool_layer([0], [1], "p3"), range(self.num_qubits // 4 * 3, self.num_qubits), inplace=True)

        return qc.decompose()

    # ---------------------------------------------------------------------
    #  Estimator‑QNN wrapper
    # ---------------------------------------------------------------------
    def _build_qnn(self) -> EstimatorQNN:
        observable = SparsePauliOp.from_list([("Z" + "I" * (self.num_qubits - 1), 1)])
        return EstimatorQNN(
            circuit=self.circuit,
            observables=observable,
            input_params=self.feature_map.parameters,
            weight_params=self.circuit.parameters,
            estimator=self.estimator,
        )

    # ---------------------------------------------------------------------
    #  Prediction / training helpers
    # ---------------------------------------------------------------------
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Return expectation values for a batch of classical data."""
        return np.array(self.qnn.predict(data))

    def train_step(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        optimizer: COBYLA | None = None,
        num_trials: int = 50,
    ) -> float:
        """Perform a single training iteration using the supplied
        optimizer.  The loss is mean‑squared error between the
        predicted expectation values and the target labels.
        """
        if optimizer is None:
            optimizer = COBYLA(maxiter=num_trials)

        def loss_function(params):
            preds = self.qnn.predict(data, param_values=params)
            return np.mean((preds - labels) ** 2)

        res = optimizer.minimize(loss_function, initial_point=self.qnn.weights)
        self.qnn.set_weights(res.x)
        return res.fun


def QCNN() -> QCNN:
    """Factory that returns a default QCNN instance."""
    return QCNN()


__all__ = ["QCNN"]
