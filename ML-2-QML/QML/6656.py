from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import StatevectorEstimator

class EstimatorQNN(QiskitEstimatorQNN):
    """
    A thin wrapper around qiskit_machine_learning.neural_networks.EstimatorQNN
    that builds a simple one‑qubit variational circuit with a Pauli‑Y observable.
    """

    def __init__(self):
        # Parameters
        data_param = Parameter("x")
        weight_param = Parameter("w")

        # Circuit
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.ry(data_param, 0)
        qc.rx(weight_param, 0)

        # Observable
        observable = SparsePauliOp.from_list([("Y", 1)])

        # Estimator
        estimator = StatevectorEstimator()

        super().__init__(
            circuit=qc,
            observables=observable,
            input_params=[data_param],
            weight_params=[weight_param],
            estimator=estimator
        )

__all__ = ["EstimatorQNN"]
