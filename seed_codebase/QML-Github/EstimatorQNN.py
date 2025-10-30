from qiskit.circuit import Parameter
from qiskit import QuantumCircuit

def EstimatorQNN():
    params1 = [Parameter("input1"), Parameter("weight1")]
    qc1 = QuantumCircuit(1)
    qc1.h(0)
    qc1.ry(params1[0], 0)
    qc1.rx(params1[1], 0)
    qc1.draw("mpl", style="clifford")

    from qiskit.quantum_info import SparsePauliOp

    observable1 = SparsePauliOp.from_list([("Y" * qc1.num_qubits, 1)])

    from qiskit_machine_learning.neural_networks import EstimatorQNN
    from qiskit.primitives import StatevectorEstimator as Estimator

    estimator = Estimator()
    estimator_qnn = EstimatorQNN(
        circuit=qc1,
        observables=observable1,
        input_params=[params1[0]],
        weight_params=[params1[1]],
        estimator=estimator,
    )
    return estimator_qnn