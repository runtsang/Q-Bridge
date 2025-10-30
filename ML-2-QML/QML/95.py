"""Quantum QCNN with noise simulation and a lightweight training helper."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.providers.aer import AerSimulator
from qiskit.providers.aer.noise import NoiseModel, depolarizing_error
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals


class QCNNModel(EstimatorQNN):
    """
    Quantum convolutional neural network with noise simulation,
    a parameter‑shift gradient placeholder, and a simple training
    helper that works with classical optimizers.
    """

    def __init__(self,
                 circuit: QuantumCircuit,
                 observable: SparsePauliOp,
                 input_params: list[ParameterVector],
                 weight_params: list[ParameterVector],
                 estimator: Estimator | None = None,
                 noise_model: NoiseModel | None = None,
                 seed: int | None = None) -> None:
        super().__init__(circuit=circuit,
                         observables=observable,
                         input_params=input_params,
                         weight_params=weight_params,
                         estimator=estimator)
        if seed is not None:
            algorithm_globals.random_seed = seed
        self.noise_model = noise_model or NoiseModel()
        self.backend = AerSimulator(noise_model=self.noise_model,
                                    seed_simulator=seed)

    def _estimator_run(self,
                       input_data: np.ndarray,
                       weight_data: np.ndarray) -> np.ndarray:
        """Send jobs to the noisy simulator."""
        return super()._estimator_run(input_data=input_data,
                                      weight_data=weight_data,
                                      backend=self.backend)

    def train(self,
              X: np.ndarray,
              y: np.ndarray,
              optimizer,
              epochs: int = 20,
              batch_size: int = 8,
              verbose: bool = True) -> None:
        """
        Very small training loop that expects an optimizer
        with ``step`` and ``zero_grad`` methods.
        """
        for epoch in range(epochs):
            for i in range(0, len(X), batch_size):
                xb = X[i:i + batch_size]
                yb = y[i:i + batch_size]

                def loss_fn():
                    preds = self(xb, w=optimizer.weights)
                    return ((preds - yb) ** 2).mean()

                optimizer.zero_grad()
                loss = loss_fn()
                loss.backward()
                optimizer.step()
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} loss: {loss.item():.4f}")


def QCNN() -> QCNNModel:
    """Build the QCNN circuit and return a QCNNModel."""
    algorithm_globals.random_seed = 12345
    estimator = Estimator()

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
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(param_prefix, length=num_qubits * 3)
        for i in range(0, num_qubits, 2):
            qc.append(conv_circuit(params[i * 3:(i + 1) * 3]), [i, i + 1])
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
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
        for i, (s, t) in enumerate(zip(sources, sinks)):
            qc.append(pool_circuit(params[i * 3:(i + 1) * 3]), [s, t])
        return qc

    feature_map = ZFeatureMap(8)
    ansatz = QuantumCircuit(8)

    # First Convolutional Layer
    ansatz.compose(conv_layer(8, "c1"), inplace=True)

    # First Pooling Layer
    ansatz.compose(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), inplace=True)

    # Second Convolutional Layer
    ansatz.compose(conv_layer(4, "c2"), inplace=True)

    # Second Pooling Layer
    ansatz.compose(pool_layer([0, 1], [2, 3], "p2"), inplace=True)

    # Third Convolutional Layer
    ansatz.compose(conv_layer(2, "c3"), inplace=True)

    # Third Pooling Layer
    ansatz.compose(pool_layer([0], [1], "p3"), inplace=True)

    # Combine feature map and ansatz
    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, inplace=True)
    circuit.compose(ansatz, inplace=True)

    observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

    # Noise model: depolarizing error on all single‑qubit gates
    noise = NoiseModel()
    depol = depolarizing_error(0.001, 1)
    for gate in ["u1", "u2", "u3", "cx"]:
        noise.add_all_qubit_quantum_error(depol, gate)

    return QCNNModel(
        circuit=circuit,
        observable=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
        noise_model=noise,
        seed=12345
    )


__all__ = ["QCNN", "QCNNModel"]
