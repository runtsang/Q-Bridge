from qiskit.circuit import ParameterVector
from qiskit import QuantumCircuit
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp

def EstimatorQNN() -> QiskitEstimatorQNN:
    """
    Three‑qubit variational estimator.

    The circuit encodes three input features via RY rotations, applies two
    layers of entanglement (CNOT ladder), and two sets of variational RZ
    rotations.  The expectation value of Pauli‑Z on qubit 0 is returned as
    the prediction.
    """
    # Parameters
    input_params = ParameterVector("x", 3)   # x0, x1, x2
    weight_params1 = ParameterVector("w1", 3)  # w0, w1, w2
    weight_params2 = ParameterVector("w2", 3)  # w0', w1', w2'

    qc = QuantumCircuit(3)

    # Data encoding: RY rotations on each qubit
    for i in range(3):
        qc.ry(input_params[i], i)

    # Variational layer 1: RZ rotations (weights)
    for i in range(3):
        qc.rz(weight_params1[i], i)

    # Entanglement layer 1
    qc.cx(0, 1)
    qc.cx(1, 2)

    # Variational layer 2: RZ rotations (weights)
    for i in range(3):
        qc.rz(weight_params2[i], i)

    # Entanglement layer 2
    qc.cx(2, 0)
    qc.cx(0, 1)

    # Measurement observable
    observable = SparsePauliOp.from_list([("Z" * qc.num_qubits, 1)])

    # Estimator primitive
    estimator = StatevectorEstimator()

    return QiskitEstimatorQNN(
        circuit=qc,
        observables=observable,
        input_params=input_params,
        weight_params=list(weight_params1) + list(weight_params2),
        estimator=estimator,
    )

__all__ = ["EstimatorQNN"]
