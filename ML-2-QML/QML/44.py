from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import Estimator as StatevectorEstimator

def EstimatorQNN(num_qubits: int = 2, depth: int = 2):
    """Hybrid variational quantum neural network.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    depth : int
        Number of variational layers.

    The circuit applies an Ry rotation per qubit for input encoding,
    followed by *depth* layers of parameterised Rx gates and linear
    entanglement via CNOTs.  Observables comprise single‑qubit Z
    operators and nearest‑neighbour ZZ strings, providing a richer
    feature set than the original single‑qubit example.
    """
    # Input and weight parameters
    input_params = [Parameter(f"x{i}") for i in range(num_qubits)]
    weight_params = [Parameter(f"w{d}_{i}") for d in range(depth) for i in range(num_qubits)]

    qc = QuantumCircuit(num_qubits)

    # Input encoding
    for i, p in enumerate(input_params):
        qc.ry(p, i)

    # Variational layers with entanglement
    for d in range(depth):
        for i in range(num_qubits):
            qc.rx(weight_params[d * num_qubits + i], i)
        # Linear chain entanglement
        for i in range(num_qubits - 1):
            qc.cx(i, i + 1)

    # Observables: Z on each qubit and ZZ on adjacent pairs
    observables = []
    for i in range(num_qubits):
        pauli_str = ["I"] * num_qubits
        pauli_str[i] = "Z"
        observables.append(SparsePauliOp.from_list([("".join(pauli_str), 1)]))
    for i in range(num_qubits - 1):
        pauli_str = ["I"] * num_qubits
        pauli_str[i] = "Z"
        pauli_str[i + 1] = "Z"
        observables.append(SparsePauliOp.from_list([("".join(pauli_str), 1)]))

    estimator = StatevectorEstimator()
    estimator_qnn = QiskitEstimatorQNN(
        circuit=qc,
        observables=observables,
        input_params=input_params,
        weight_params=weight_params,
        estimator=estimator,
    )
    return estimator_qnn
