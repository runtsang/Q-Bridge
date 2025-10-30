"""Quantum variant of EstimatorQNNGen197.

The circuit encodes a 4‑dimensional classical feature vector into a 4‑qubit register
via parameterised RY rotations.  A lightweight variational layer with trainable
RX, RY, RZ and CRX gates introduces learnable weights.  The expectation value of
the Y Pauli operator on qubit 0 is used as the regression output.  The model is
instantiated through Qiskit’s EstimatorQNN wrapper and runs on the StatevectorEstimator.

The design links the classical feature extractor (the 4‑dimensional embedding produced
by EstimatorQNNGen197’s FC head) directly to the quantum circuit, enabling a hybrid
classical‑quantum training pipeline.
"""
from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator


def EstimatorQNNGen197() -> EstimatorQNN:
    # Input parameters: 4 classical features
    input_params = [Parameter(f"input_{i}") for i in range(4)]
    # Trainable weight parameters for the variational layer
    weight_params = [
        Parameter("w_rx0"),
        Parameter("w_ry1"),
        Parameter("w_rz2"),
        Parameter("w_crx3"),
    ]

    qc = QuantumCircuit(4)

    # Feature encoding: RY(input_i) on qubit i
    for i, p in enumerate(input_params):
        qc.ry(p, i)

    # Variational layer
    qc.rx(weight_params[0], 0)
    qc.ry(weight_params[1], 1)
    qc.rz(weight_params[2], 2)
    qc.crx(weight_params[3], 0, 3)  # controlled rotation with qubit 3

    # Measurement observable: Y on qubit 0 (Pauli string "YIII")
    observable = SparsePauliOp.from_list([("Y" + "I" * (qc.num_qubits - 1), 1)])

    estimator = StatevectorEstimator()

    return EstimatorQNN(
        circuit=qc,
        observables=observable,
        input_params=input_params,
        weight_params=weight_params,
        estimator=estimator,
    )


__all__ = ["EstimatorQNNGen197"]
