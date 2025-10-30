"""Quantum QCNN with a deeper ansatz and SPSA training."""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit.algorithms.optimizers import SPSA
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals


class QCNNGen321QNN(EstimatorQNN):
    """QCNN ansatz with residual entanglement and custom SPSA training."""

    def __init__(self, *, feature_map: QuantumCircuit, ansatz: QuantumCircuit,
                 observable: SparsePauliOp, estimator: Estimator):
        super().__init__(
            circuit=ansatz.decompose(),
            observables=observable,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            estimator=estimator
        )

    def train(self, X, y, maxiter: int = 200, step: float = 0.01):
        """Train the QNN using SPSA optimizer."""
        optimizer = SPSA(maxiter=maxiter, step_size=step)
        init_guess = np.random.uniform(-np.pi, np.pi, len(self.weight_params))

        def loss_fn(params):
            return self.loss(params, X, y)

        opt_params, _ = optimizer.optimize(
            num_vars=len(init_guess),
            objective_function=loss_fn,
            initial_point=init_guess
        )
        self.set_weights(opt_params)
        return opt_params


def QCNN() -> QCNNGen321QNN:
    """Factory returning a QCNNGen321QNN instance."""
    algorithm_globals.random_seed = 12345
    estimator = Estimator()

    # Feature map
    feature_map = ZFeatureMap(8)

    # Convolutional sub‑circuit
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

    def conv_layer(num_qubits, prefix):
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(prefix, length=num_qubits * 3 // 2)
        idx = 0
        for i in range(0, num_qubits, 2):
            sub = conv_circuit(params[idx : idx + 3])
            qc.append(sub, [i, i + 1])
            idx += 3
        return qc

    # Pooling sub‑circuit
    def pool_circuit(params):
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def pool_layer(sources, sinks, prefix):
        num = len(sources) + len(sinks)
        qc = QuantumCircuit(num)
        params = ParameterVector(prefix, length=num // 2 * 3)
        idx = 0
        for s, t in zip(sources, sinks):
            sub = pool_circuit(params[idx : idx + 3])
            qc.append(sub, [s, t])
            idx += 3
        return qc

    # Build the full ansatz
    ansatz = QuantumCircuit(8)
    ansatz.compose(conv_layer(8, "c1"), range(8), inplace=True)
    ansatz.compose(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), range(8), inplace=True)
    ansatz.compose(conv_layer(4, "c2"), range(4, 8), inplace=True)
    ansatz.compose(pool_layer([0, 1], [2, 3], "p2"), range(4, 8), inplace=True)
    ansatz.compose(conv_layer(2, "c3"), range(6, 8), inplace=True)
    ansatz.compose(pool_layer([0], [1], "p3"), range(6, 8), inplace=True)

    # Observable
    observable = SparsePauliOp.from_list([("Z" * 8, 1)])

    qnn = QCNNGen321QNN(
        feature_map=feature_map,
        ansatz=ansatz,
        observable=observable,
        estimator=estimator
    )
    return qnn


__all__ = ["QCNN", "QCNNGen321QNN"]
