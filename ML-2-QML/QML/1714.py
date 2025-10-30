from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp

def EstimatorQNN():
    """Return a Qiskit EstimatorQNN with a 2‑qubit parameter‑shift circuit."""
    # Parameters for input encoding and trainable weights
    input_params = [Parameter("x1"), Parameter("x2")]
    weight_params = [Parameter("w1"), Parameter("w2")]

    # Build a simple 2‑qubit circuit
    qc = QuantumCircuit(2)
    # Input encoding: rotations on each qubit
    qc.ry(input_params[0], 0)
    qc.ry(input_params[1], 1)
    # Parameter‑shift layer
    qc.rx(weight_params[0], 0)
    qc.rz(weight_params[1], 1)
    qc.cx(0, 1)

    # Observable: Pauli‑X on qubit 0 ⊗ Pauli‑Y on qubit 1
    observable = SparsePauliOp.from_list([("XY", 1)])

    # Instantiate the EstimatorQNN
    estimator = StatevectorEstimator()
    estimator_qnn = QiskitEstimatorQNN(
        circuit=qc,
        observables=observable,
        input_params=input_params,
        weight_params=weight_params,
        estimator=estimator,
    )
    return estimator_qnn

__all__ = ["EstimatorQNN"]
