from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import StatevectorEstimator

def EstimatorQNN() -> QiskitEstimatorQNN:
    """
    Quantum neural network that maps a 4‑dimensional classical feature vector
    to a regression output using a parameterised variational circuit.
    The circuit combines input‑encoding rotations with a depth‑2 entangling
    layer, and the observable is the sum of Pauli‑Z on all qubits.
    """
    # Parameters: 4 for the classical inputs, 8 for the trainable weights
    input_params = [Parameter(f"x{i}") for i in range(4)]
    weight_params = [Parameter(f"w{i}") for i in range(8)]

    qc = QuantumCircuit(4)

    # Input encoding: Ry rotations
    for i, p in enumerate(input_params):
        qc.ry(p, i)

    # Weight rotations (first set)
    for i, p in enumerate(weight_params[:4]):
        qc.ry(p, i)

    # Entangling layer (depth‑2)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)
    qc.cx(3, 0)

    # Second set of weight rotations
    for i, p in enumerate(weight_params[4:], start=0):
        qc.ry(p, i)

    # Observable: sum of Pauli‑Z on all qubits
    observable = SparsePauliOp.from_list([("Z" * 4, 1)])

    # Estimator primitive
    estimator = StatevectorEstimator()

    # Construct the Qiskit EstimatorQNN
    estimator_qnn = QiskitEstimatorQNN(
        circuit=qc,
        observables=observable,
        input_params=input_params,
        weight_params=weight_params,
        estimator=estimator,
    )
    return estimator_qnn

__all__ = ["EstimatorQNN"]
