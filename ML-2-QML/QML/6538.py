from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import Estimator as StatevectorEstimator

def EstimatorQNN():
    """Return a two‑qubit variational quantum neural network with data re‑uploading."""
    # Input parameters representing features
    input_params = [Parameter("x1"), Parameter("x2")]

    # Trainable rotation parameters
    weight_params = [Parameter(f"w{i}") for i in range(8)]

    qc = QuantumCircuit(2)

    # Data re‑uploading layers
    qc.ry(input_params[0], 0)
    qc.rz(input_params[1], 1)

    # Entanglement and rotation layers
    for i in range(4):
        qc.ry(weight_params[2 * i], 0)
        qc.rz(weight_params[2 * i + 1], 1)
        qc.cx(0, 1)

    qc.barrier()

    # Observable for regression output
    observable = SparsePauliOp.from_list([("YZ", 1)])

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
