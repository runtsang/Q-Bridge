"""
QCNNEnhanced: Quantum kernel QCNN wrapped in a PyTorch‑style class.

The class exposes a ``forward`` method that evaluates the
variational circuit and a ``fit`` method that optimizes the
ansatz parameters with a COBYLA optimizer.  The circuit
builds upon the original seed but replaces the fixed
convolutional layers with a learnable variational kernel
parameterised by the same 3‑qubit blocks.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals

class QCNNModel:
    """
    Quantum QCNN implemented as a QNN wrapped in a class
    compatible with the classical ``QCNNModel`` interface.
    """
    def __init__(self) -> None:
        # Shared random seed for reproducibility
        algorithm_globals.random_seed = 12345
        estimator = StatevectorEstimator()
        # Build the feature map
        self.feature_map = ZFeatureMap(8)
        # Build the learnable ansatz
        self.ansatz = self._build_ansatz()
        # Observable for binary classification
        self.observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])
        # Construct EstimatorQNN
        self.qnn = EstimatorQNN(
            circuit=self.ansatz.decompose(),
            observables=self.observable,
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            estimator=estimator,
        )
        # Store parameters for optimisation
        self.params = self.ansatz.parameters

    def _build_ansatz(self) -> QuantumCircuit:
        """
        Construct a variational ansatz using the 3‑qubit building block
        from the seed.  The ansatz contains three convolutional layers
        followed by three pooling layers, mirroring the original QCNN.
        """
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

        def conv_layer(num_qubits, param_prefix):
            qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
            qubits = list(range(num_qubits))
            param_index = 0
            params = ParameterVector(param_prefix, length=num_qubits * 3)
            for q1, q2 in zip(qubits[0::2], qubits[1::2]):
                qc.compose(conv_circuit(params[param_index:param_index+3]), [q1, q2], inplace=True)
                qc.barrier()
                param_index += 3
            for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
                qc.compose(conv_circuit(params[param_index:param_index+3]), [q1, q2], inplace=True)
                qc.barrier()
                param_index += 3
            return qc

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
                qc.compose(pool_circuit(params[param_index:param_index+3]), [source, sink], inplace=True)
                qc.barrier()
                param_index += 3
            return qc

        # Assemble the full ansatz
        ansatz = QuantumCircuit(8, name="QCNN Ansatz")
        ansatz.compose(conv_layer(8, "c1"), range(8), inplace=True)
        ansatz.compose(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), range(8), inplace=True)
        ansatz.compose(conv_layer(4, "c2"), range(4, 8), inplace=True)
        ansatz.compose(pool_layer([0, 1], [2, 3], "p2"), range(4, 8), inplace=True)
        ansatz.compose(conv_layer(2, "c3"), range(6, 8), inplace=True)
        ansatz.compose(pool_layer([0], [1], "p3"), range(6, 8), inplace=True)
        return ansatz

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Evaluate the QNN on a batch of classical inputs.

        Parameters
        ----------
        inputs : np.ndarray
            Array of shape (n_samples, 8) representing the feature vectors.

        Returns
        -------
        np.ndarray
            Probabilities of shape (n_samples, 1).
        """
        return self.qnn(inputs)

    def fit(self, X: np.ndarray, y: np.ndarray,
            epochs: int = 10, lr: float = 1e-3) -> None:
        """
        Train the quantum ansatz parameters using COBYLA.

        Parameters
        ----------
        X : np.ndarray
            Training data of shape (n_samples, 8).
        y : np.ndarray
            Binary labels of shape (n_samples, 1).
        epochs : int, optional
            Number of optimisation iterations.
        lr : float, optional
            Not used by COBYLA but kept for API symmetry.
        """
        optimizer = COBYLA(maxiter=epochs * 10)
        # Convert labels to probabilities
        y = y.astype(np.float64).reshape(-1, 1)

        def loss_fn(params):
            # Bind new parameters to the ansatz
            bound_qc = self.ansatz.bind_parameters(
                {p: val for p, val in zip(self.ansatz.parameters, params)}
            )
            # Build a new EstimatorQNN with bound parameters
            qnn = EstimatorQNN(
                circuit=bound_qc.decompose(),
                observables=self.observable,
                input_params=self.feature_map.parameters,
                weight_params=self.ansatz.parameters,
                estimator=StatevectorEstimator(),
            )
            preds = qnn(X)
            # Binary cross‑entropy loss
            loss = -np.mean(y * np.log(preds + 1e-12) + (1 - y) * np.log(1 - preds + 1e-12))
            return loss

        # Initialise with current parameters
        init_params = np.array([self.ansatz.parameters[i].value for i in range(len(self.ansatz.parameters))])
        # Run optimisation
        result = optimizer.minimize(loss_fn, init_params)
        # Update ansatz with optimized parameters
        for i, p in enumerate(self.ansatz.parameters):
            p.assign_value(result.x[i])

def QCNN() -> QCNNModel:
    """
    Factory returning an instance of the quantum QCNN.
    """
    return QCNNModel()

__all__ = ["QCNN", "QCNNModel"]
