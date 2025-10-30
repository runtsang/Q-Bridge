"""Hybrid quantum neural network with 3‑qubit entangled circuit and multiple observables."""

from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import Estimator as StatevectorEstimator

def EstimatorQNN() -> QiskitEstimatorQNN:
    """Return a Qiskit EstimatorQNN with a 3‑qubit entangled feature map."""
    # Define parameters
    input_params = [Parameter(f"x{i}") for i in range(3)]
    weight_params = [Parameter(f"w{i}") for i in range(6)]  # 2 trainable per qubit

    qc = QuantumCircuit(3)

    # Feature map: Ry rotations with input data
    for i, p in enumerate(input_params):
        qc.ry(p, i)

    # Entanglement: CNOT chain
    qc.cx(0, 1)
    qc.cx(1, 2)

    # Trainable rotations
    qc.rx(weight_params[0], 0)
    qc.ry(weight_params[1], 0)
    qc.rx(weight_params[2], 1)
    qc.ry(weight_params[3], 1)
    qc.rx(weight_params[4], 2)
    qc.ry(weight_params[5], 2)

    # Observables: Pauli Y on each qubit
    observables = SparsePauliOp.from_list([("Y" * qc.num_qubits, 1)])

    # Estimator backend
    estimator = StatevectorEstimator()

    # Construct EstimatorQNN
    estimator_qnn = QiskitEstimatorQNN(
        circuit=qc,
        observables=observables,
        input_params=input_params,
        weight_params=weight_params,
        estimator=estimator,
    )
    return estimator_qnn

__all__ = ["EstimatorQNN"]
