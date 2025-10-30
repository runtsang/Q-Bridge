"""Quantum QCNN hybrid model with parameterised ansatz and adaptive pooling.

The quantum architecture extends the original QCNN by:
* Using a modular entanglement block that can be reused across convolutional layers.
* Supporting adaptive pooling that can be turned off for ablation studies.
* Allowing multi‑observable readout to enable richer classification signals.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator as QuantumEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.optimizers import Adam
from qiskit.utils import QuantumInstance
from qiskit.providers.fake_provider import FakeVigo


class QCNNHybrid:
    """Quantum convolutional neural network ansatz with adaptive pooling.

    The model is built from a feature map, a stack of convolutional blocks that
    apply a two‑qubit entangling unitary, and optional pooling layers.  The
    ansatz is wrapped in an :class:`EstimatorQNN` for efficient gradient
    evaluation on a backend simulator or real device.
    """

    def __init__(
        self,
        n_qubits: int = 8,
        conv_depth: int = 3,
        pool_depth: int = 3,
        use_pooling: bool = True,
        backend=None,
        seed: int | None = 42,
    ) -> None:
        self.n_qubits = n_qubits
        self.conv_depth = conv_depth
        self.pool_depth = pool_depth
        self.use_pooling = use_pooling
        self.backend = backend or FakeVigo().backend()
        self.seed = seed

        # Feature map
        self.feature_map = ZFeatureMap(num_qubits=n_qubits, reps=1, entanglement='full')

        # Ansatz construction
        self.ansatz = self._build_ansatz()

        # Observables: single Z on first qubit for binary classification
        self.observable = SparsePauliOp.from_list([("Z" + "I" * (n_qubits - 1), 1.0)])

        # EstimatorQNN
        self.estimator = QuantumEstimator(quantum_instance=QuantumInstance(self.backend, seed_simulator=self.seed))
        self.qnn = EstimatorQNN(
            circuit=self.ansatz,
            observables=self.observable,
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
            estimator=self.estimator,
        )

        # Optimizer
        self.optimizer = Adam(learning_rate=0.01)

    def _entanglement_block(self, params: ParameterVector) -> QuantumCircuit:
        """Two‑qubit entangling block used in convolution layers."""
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.ry(params[0], 0)
        qc.rz(params[1], 1)
        qc.cx(1, 0)
        return qc

    def _conv_layer(self, qubit_indices: list[int], param_prefix: str) -> QuantumCircuit:
        """Apply the entanglement block across adjacent qubit pairs."""
        qc = QuantumCircuit(self.n_qubits)
        params = ParameterVector(param_prefix, length=len(qubit_indices) // 2 * 2)
        idx = 0
        for q1, q2 in zip(qubit_indices[0::2], qubit_indices[1::2]):
            block = self._entanglement_block(params[idx : idx + 2])
            qc.append(block, [q1, q2])
            idx += 2
        return qc

    def _pool_layer(self, sources: list[int], sinks: list[int], param_prefix: str) -> QuantumCircuit:
        """Simple pooling that measures a qubit and discards it."""
        qc = QuantumCircuit(self.n_qubits)
        params = ParameterVector(param_prefix, length=len(sources) * 2)
        idx = 0
        for src, sink in zip(sources, sinks):
            block = self._entanglement_block(params[idx : idx + 2])
            qc.append(block, [src, sink])
            idx += 2
        return qc

    def _build_ansatz(self) -> QuantumCircuit:
        """Compose the full ansatz with convolution and optional pooling."""
        qc = QuantumCircuit(self.n_qubits)
        # Convolution layers
        for d in range(self.conv_depth):
            qc.compose(self._conv_layer(list(range(self.n_qubits)), f"c{d}"), inplace=True)
            if self.use_pooling:
                # Pooling reduces qubits by half
                new_qubits = list(range(self.n_qubits // (2 ** (d + 1))))
                qc.compose(self._pool_layer(list(range(self.n_qubits // (2 ** d))), new_qubits, f"p{d}"), inplace=True)
        return qc

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return probability estimates for binary classification."""
        preds = self.qnn.predict(features)
        return preds.squeeze()

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 10,
        batch_size: int = 32,
        verbose: bool = True,
    ) -> None:
        """Train the quantum network using gradient descent."""
        dataset = list(zip(X, y))
        for epoch in range(epochs):
            np.random.shuffle(dataset)
            losses = []
            for i in range(0, len(dataset), batch_size):
                batch = dataset[i : i + batch_size]
                X_batch, y_batch = np.vstack([b[0] for b in batch]), np.vstack([b[1] for b in batch])
                loss = self.qnn.train_step(X_batch, y_batch, optimizer=self.optimizer)
                losses.append(loss)
            if verbose:
                print(f"Epoch {epoch + 1}/{epochs} | Loss: {np.mean(losses):.4f}")

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return classification accuracy."""
        preds = self.predict(X) > 0.5
        return (preds == y).mean()


__all__ = ["QCNNHybrid"]
