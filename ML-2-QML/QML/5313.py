from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator

def HybridQuantumEstimator():
    """
    Builds a Qiskit EstimatorQNN that mirrors the quantum part of
    HybridEstimatorQNN.  The circuit contains a single qubit with a
    parameterized input rotation followed by two trainable rotations.
    The observable is the Pauli‑Y operator.
    """
    # Define parameters
    input_param = Parameter('x')
    w1 = Parameter('w1')
    w2 = Parameter('w2')

    # Circuit
    qc = QuantumCircuit(1)
    qc.ry(input_param, 0)
    qc.ry(w1, 0)
    qc.rz(w2, 0)

    # Observable
    observable = SparsePauliOp.from_list([('Y', 1)])

    # Estimator
    estimator = StatevectorEstimator()

    return EstimatorQNN(circuit=qc,
                        observables=observable,
                        input_params=[input_param],
                        weight_params=[w1, w2],
                        estimator=estimator)
