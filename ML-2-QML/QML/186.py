"""Quantum QCNN implementation with parameter‑shift gradients and a simple SGD trainer."""

from __future__ import annotations

import numpy as np
from typing import Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.optimizers import Adam
from qiskit.utils import algorithm_globals


class QCNNModel:
    """
    Quantum QCNN wrapper that mirrors the classical QCNNModel API.

    Parameters
    ----------
    circuit : QuantumCircuit
        Full QCNN circuit (feature map + ansatz).
    observables : SparsePauliOp
        Observable(s) for the expectation value.
    estimator : Estimator
        Backend estimator used for state‑vector or hardware execution.
    input_params : Sequence[Parameter]
        Parameters of the feature map.
    weight_params : Sequence[Parameter]
        Parameters of the ansatz.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        observables: SparsePauliOp,
        estimator: Estimator,
        input_params: Sequence,
        weight_params: Sequence,
    ) -> None:
        self.circuit = circuit
        self.observables = observables
        self.estimator = estimator
        self.input_params = input_params
        self.weight_params = weight_params

        self.qnn = EstimatorQNN(
            circuit=circuit.decompose(),
            observables=observables,
            input_params=input_params,
            weight_params=weight_params,
            estimator=estimator,
        )

        # initialise weights uniformly in [-π, π]
        self.params = np.random.uniform(-np.pi, np.pi, len(weight_params))

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Evaluate the QCNN on a batch of classical data.

        Parameters
        ----------
        inputs : np.ndarray
            Shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Expectation values of shape (n_samples,).
        """
        return self.qnn(inputs, weight_params=self.params)

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """Alias for :meth:`forward`."""
        return self.forward(inputs)

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        lr: float = 0.01,
        optimizer: str = "adam",
        verbose: bool = True,
    ) -> None:
        """
        Train the QCNN with a simple stochastic gradient descent loop.

        Parameters
        ----------
        X : np.ndarray
            Training inputs (n_samples, n_features).
        y : np.ndarray
            Binary labels (n_samples,).
        epochs : int
            Number of epochs.
        lr : float
            Learning rate.
        optimizer : str
            Optimizer to use ('adam' or'sgd').
        verbose : bool
            Print progress.
        """
        if optimizer == "adam":
            opt = Adam(lr)
        else:
            opt = None  # SGD via manual update

        for epoch in range(1, epochs + 1):
            # Forward pass
            preds = self.forward(X).flatten()
            loss = np.mean((preds - y) ** 2)

            # Gradient via parameter‑shift
            grads = self.estimator.gradient(
                np.concatenate((self.input_params, self.params)),
                X,
                self.observables,
            )

            # Extract weight gradients
            weight_grads = grads[-len(self.params) :]

            # Update weights
            if optimizer == "adam":
                self.params = opt.update(self.params, weight_grads)
            else:
                self.params -= lr * weight_grads

            if verbose and epoch % max(1, epochs // 10) == 0:
                print(f"Epoch {epoch}/{epochs} – Loss: {loss:.6f}")

    def get_parameters(self) -> np.ndarray:
        """Return current weight parameters."""
        return self.params

    def set_parameters(self, params: np.ndarray) -> None:
        """Set new weight parameters."""
        self.params = params

    def get_circuit(self) -> QuantumCircuit:
        """Return the underlying quantum circuit."""
        return self.circuit

def QCNN() -> QCNNModel:
    """
    Construct and return a QCNNModel instance using the standard 8‑qubit
    architecture described in the original seed.
    """
    algorithm_globals.random_seed = 12345
    estimator = Estimator()

    # Convolution and pooling sub‑circuits (identical to the seed)
    def conv_circuit(params: ParameterVector) -> QuantumCircuit:
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

    def conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits)
        qubits = list(range(num_qubits))
        params = ParameterVector(param_prefix, length=num_qubits * 3 // 2)
        idx = 0
        for q1, q2 in zip(qubits[0::2], qubits[1::2]):
            qc.compose(conv_circuit(params[idx : idx + 3]), [q1, q2], inplace=True)
            qc.barrier()
            idx += 3
        return qc

    def pool_circuit(params: ParameterVector) -> QuantumCircuit:
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def pool_layer(sources: Sequence[int], sinks: Sequence[int], param_prefix: str) -> QuantumCircuit:
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(param_prefix, length=len(sources) * 3)
        idx = 0
        for src, sink in zip(sources, sinks):
            qc.compose(pool_circuit(params[idx : idx + 3]), [src, sink], inplace=True)
            qc.barrier()
            idx += 3
        return qc

    # Build the full ansatz
    ansatz = QuantumCircuit(8, name="Ansatz")
    ansatz.compose(conv_layer(8, "c1"), inplace=True)
    ansatz.compose(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), inplace=True)
    ansatz.compose(conv_layer(4, "c2"), inplace=True)
    ansatz.compose(pool_layer([0, 1], [2, 3], "p2"), inplace=True)
    ansatz.compose(conv_layer(2, "c3"), inplace=True)
    ansatz.compose(pool_layer([0], [1], "p3"), inplace=True)

    # Feature map
    feature_map = ZFeatureMap(8)

    # Full circuit
    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, range(8), inplace=True)
    circuit.compose(ansatz, range(8), inplace=True)

    observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

    return QCNNModel(
        circuit=circuit,
        observables=observable,
        estimator=estimator,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
    )

__all__ = ["QCNN", "QCNNModel"]
