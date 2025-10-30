from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import Estimator

def EstimatorQNN():
    """Return a 2-qubit variational estimator with entanglement and multiple trainable parameters."""
    # Input parameters (features)
    input_params = [Parameter("x0"), Parameter("x1")]
    # Weight parameters (trainable)
    weight_params = [Parameter(f"w{i}") for i in range(6)]  # 6 trainable params

    qc = QuantumCircuit(2)
    # Encode inputs
    qc.ry(input_params[0], 0)
    qc.ry(input_params[1], 1)
    # Variational layers with entanglement
    for i in range(3):
        qc.ry(weight_params[2 * i], 0)
        qc.rz(weight_params[2 * i + 1], 1)
        qc.cx(0, 1)

    # Observable: Pauli Z on both qubits (ZZ)
    observables = [SparsePauliOp.from_list([("ZZ", 1)])]

    # Estimator backend
    estimator = Estimator()

    # Construct the EstimatorQNN
    estimator_qnn = QiskitEstimatorQNN(
        circuit=qc,
        observables=observables,
        input_params=input_params,
        weight_params=weight_params,
        estimator=estimator,
    )
    return estimator_qnn

__all__ = ["EstimatorQNN"]
