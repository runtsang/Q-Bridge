"""Quantum neural network for regression using Qiskit.

The circuit employs two qubits, with input‑encoding Ry rotations, an entanglement layer,
trainable weight rotations, and a second entanglement layer. The output is the
expectation value of the observable Y⊗Z, computed via a StatevectorEstimator.
"""

from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator


def EstimatorQNN() -> EstimatorQNN:
    """
    Build a small yet expressive QNN:

        - 2 qubits
        - Input encoding via Ry rotations on each qubit
        - Entanglement layer using CNOTs
        - Trainable weight rotations on each qubit
        - Entanglement layer again
        - Measure expectation of Y⊗Z

    Returns:
        qiskit_machine_learning.neural_networks.EstimatorQNN instance ready for training.
    """
    # Define parameters
    input_params = [Parameter(f"input_{i}") for i in range(2)]
    weight_params = [Parameter(f"weight_{i}") for i in range(2)]

    qc = QuantumCircuit(2)

    # Input encoding: Ry(input_i) on qubit i
    qc.ry(input_params[0], 0)
    qc.ry(input_params[1], 1)

    # Entanglement layer
    qc.cx(0, 1)
    qc.cx(1, 0)

    # Weight rotations
    qc.ry(weight_params[0], 0)
    qc.ry(weight_params[1], 1)

    # Second entanglement
    qc.cx(0, 1)
    qc.cx(1, 0)

    # Observable: Y⊗Z expectation value
    observable = SparsePauliOp.from_list([("YZ", 1)])

    # Backend estimator
    estimator = StatevectorEstimator()

    # Construct the QNN
    qnn = EstimatorQNN(
        circuit=qc,
        observables=observable,
        input_params=input_params,
        weight_params=weight_params,
        estimator=estimator,
    )
    return qnn


__all__ = ["EstimatorQNN"]
