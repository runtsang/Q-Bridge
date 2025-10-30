from qiskit.circuit import ParameterVector
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator

def HybridEstimatorQNN(num_qubits: int = 4, input_dim: int = 2) -> EstimatorQNN:
    """Quantum estimator that mirrors the classical HybridEstimatorQNN.

    The circuit encodes ``input_dim`` classical features into the first qubits via
    Ry rotations, then applies a trainable layer of RX, RY, RZ and a CNOT
    entanglement pattern.  The output is the expectation of Pauli‑Z on each qubit.
    """

    # Parameters for input encoding
    input_params = ParameterVector("x", input_dim)
    # Parameters for trainable weights
    weight_params = ParameterVector("w", num_qubits * 3)  # RX, RY, RZ per qubit

    qc = QuantumCircuit(num_qubits)

    # Initial Hadamard to create superposition
    qc.h(range(num_qubits))

    # Encode inputs on the first qubits
    for i, p in enumerate(input_params):
        qc.ry(p, i)

    # Trainable rotation layer
    for i in range(num_qubits):
        qc.rx(weight_params[3 * i], i)
        qc.ry(weight_params[3 * i + 1], i)
        qc.rz(weight_params[3 * i + 2], i)

    # Entanglement
    for i in range(num_qubits - 1):
        qc.cx(i, i + 1)
    # Wrap‑around entanglement for symmetry
    qc.cx(num_qubits - 1, 0)

    # Observables: PauliZ on each qubit
    observables = SparsePauliOp.from_list([("Z" * num_qubits, 1)])

    estimator = StatevectorEstimator()

    return EstimatorQNN(
        circuit=qc,
        observables=observables,
        input_params=list(input_params),
        weight_params=list(weight_params),
        estimator=estimator,
    )

__all__ = ["HybridEstimatorQNN"]
