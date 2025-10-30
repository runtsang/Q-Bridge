"""Quantum estimator for the hybrid model."""
from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QEstimator
from qiskit.primitives import StatevectorEstimator


def EstimatorQNN():
    """Return a quantum circuit estimator with a single input and trainable weight."""
    input_params = [Parameter("x0")]
    weight_params = [Parameter("w0")]
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.ry(input_params[0], 0)
    qc.rx(weight_params[0], 0)
    observable = SparsePauliOp.from_list([("Y", 1)])
    estimator = StatevectorEstimator()
    return QEstimator(
        circuit=qc,
        observables=observable,
        input_params=input_params,
        weight_params=weight_params,
        estimator=estimator,
    )
