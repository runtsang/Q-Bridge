"""EstimatorQNNGen342: a variational quantum circuit estimator.

Features:
* 4‑qubit hardware‑efficient ansatz with alternating RX/RZ layers.
* Entangling CNOT ladder between adjacent qubits.
* Expectation value of a Pauli‑Z tensor product observable.
* Uses Qiskit StatevectorEstimator for exact evaluation.
"""

from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator

def EstimatorQNNGen342() -> EstimatorQNN:
    # Classical input parameters
    input_params = [Parameter(f"input_{i}") for i in range(4)]
    # Variational weight parameters
    weight_params = [Parameter(f"weight_{i}") for i in range(4)]

    qc = QuantumCircuit(4)

    # Input encoding via RX rotations
    for i, p in enumerate(input_params):
        qc.rx(p, i)

    # Entangling CNOT ladder
    for i in range(3):
        qc.cx(i, i + 1)

    # Variational layer with RX and RZ rotations
    for i, p in enumerate(weight_params):
        qc.rz(p, i)
        qc.rx(p, i)

    # Observable: tensor product of Pauli‑Z on all qubits
    observable = SparsePauliOp.from_list([("Z" * qc.num_qubits, 1)])

    estimator = StatevectorEstimator()

    return EstimatorQNN(
        circuit=qc,
        observables=observable,
        input_params=input_params,
        weight_params=weight_params,
        estimator=estimator,
    )

__all__ = ["EstimatorQNNGen342"]
