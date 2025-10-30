"""Quantum convolutional neural network with extended ansatz and training utilities."""

import numpy as np
import torch
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.optimizers import L_BFGS_B
from qiskit_machine_learning.utils import algorithm_globals

def QCNN(
    num_qubits: int = 8,
    conv_depth: int = 3,
    pool_depth: int = 3,
    feature_map_depth: int = 2,
) -> EstimatorQNN:
    """
    Build a QCNN with configurable depth and feature‑map complexity.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the ansatz.
    conv_depth : int
        Number of repetitions of the two‑qubit convolution block per layer.
    pool_depth : int
        Number of repetitions of the two‑qubit pooling block per layer.
    feature_map_depth : int
        Number of repetitions of the Z‑feature map.
    """
    algorithm_globals.random_seed = 12345
    estimator = Estimator()

    def conv_circuit(params):
        target = QuantumCircuit(2)
        target.rz(-np.pi / 2, 1)
        target.cx(1, 0)
        target.rz(params[0], 0)
        target.ry(params[1], 1)
        target.cx(0, 1)
        target.ry(params[2], 1)
        target.cx(1, 0)
        target.rz(np.pi / 2, 0)
        return target

    def conv_layer(num_qubits, param_prefix, depth):
        qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
        qubits = list(range(num_qubits))
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits * 3 * depth)
        for _ in range(depth):
            for q1, q2 in zip(qubits[0::2], qubits[1::2]):
                qc = qc.compose(conv_circuit(params[param_index:param_index+3]), [q1, q2])
                qc.barrier()
                param_index += 3
            for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
                qc = qc.compose(conv_circuit(params[param_index:param_index+3]), [q1, q2])
                qc.barrier()
                param_index += 3
        qc_inst = qc.to_instruction()
        qc = QuantumCircuit(num_qubits)
        qc.append(qc_inst, qubits)
        return qc

    def pool_circuit(params):
        target = QuantumCircuit(2)
        target.rz(-np.pi / 2, 1)
        target.cx(1, 0)
        target.rz(params[0], 0)
        target.ry(params[1], 1)
        target.cx(0, 1)
        target.ry(params[2], 1)
        return target

    def pool_layer(sources, sinks, param_prefix, depth):
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits, name="Pooling Layer")
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits // 2 * 3 * depth)
        for _ in range(depth):
            for source, sink in zip(sources, sinks):
                qc = qc.compose(pool_circuit(params[param_index:param_index+3]), [source, sink])
                qc.barrier()
                param_index += 3
        qc_inst = qc.to_instruction()
        qc = QuantumCircuit(num_qubits)
        qc.append(qc_inst, range(num_qubits))
        return qc

    feature_map = ZFeatureMap(num_qubits, reps=feature_map_depth)
    obs = SparsePauliOp.from_list([("Z" + "I" * (num_qubits - 1), 1)])

    ansatz = QuantumCircuit(num_qubits, name="Ansatz")
    # Build alternating convolution / pooling layers until a single qubit remains
    remaining = num_qubits
    layer_id = 1
    while remaining > 1:
        conv_inst = conv_layer(remaining, f"c{layer_id}", conv_depth)
        ansatz.compose(conv_inst, inplace=True)
        sources = list(range(remaining // 2))
        sinks = list(range(remaining // 2, remaining))
        pool_inst = pool_layer(sources, sinks, f"p{layer_id}", pool_depth)
        ansatz.compose(pool_inst, inplace=True)
        remaining = len(sinks)
        layer_id += 1

    circuit = QuantumCircuit(num_qubits)
    circuit.compose(feature_map, range(num_qubits), inplace=True)
    circuit.compose(ansatz, range(num_qubits), inplace=True)

    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=obs,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )
    return qnn

def train_qcnn(
    qnn: EstimatorQNN,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 50,
    lr: float = 0.01,
) -> None:
    """
    Simple training loop using the L‑BFGS‑B optimiser from Qiskit.

    Parameters
    ----------
    qnn : EstimatorQNN
        The quantum neural network to train.
    X : np.ndarray
        Training inputs.
    y : np.ndarray
        Binary labels.
    epochs : int
        Number of optimisation iterations.
    lr : float
        Learning rate for the classical optimiser (ignored by L‑BFGS‑B but kept for API compatibility).
    """
    optimizer = L_BFGS_B()
    for epoch in range(epochs):
        def loss_fn(params):
            preds = qnn.predict(X, params=params)
            loss = np.mean((preds - y) ** 2)
            return loss

        params, f_val = optimizer.optimize(
            num_vars=len(qnn.weight_params),
            objective_function=loss_fn,
            initial_point=np.random.randn(len(qnn.weight_params)),
        )
        qnn.weight_params = params
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: loss = {f_val:.6f}")

__all__ = ["QCNN", "train_qcnn"]
