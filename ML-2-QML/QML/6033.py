import numpy as np
from qiskit import QuantumCircuit, Parameter
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

def quantum_conv_circuit(kernel_size: int) -> QuantumCircuit:
    """
    Build a 2-qubit convolution circuit for a kernel of size kernel_size.
    Mirrors the conv_circuit from the QML seed.
    """
    num_qubits = kernel_size ** 2
    qc = QuantumCircuit(num_qubits)
    theta = [Parameter(f"θ_{i}") for i in range(num_qubits)]
    # Apply RX rotations based on input data
    for i, t in enumerate(theta):
        qc.rx(t, i)
    # Add entangling operations (CX, RZ, etc.) similar to the seed
    for i in range(0, num_qubits - 1, 2):
        qc.cx(i, i + 1)
        qc.rz(theta[i], i)
        qc.ry(theta[i + 1], i + 1)
    return qc

def quantum_pool_circuit() -> QuantumCircuit:
    """
    Build a simple 2-qubit pooling circuit.
    Mirrors the pool_circuit from the QML seed.
    """
    qc = QuantumCircuit(2)
    theta = [Parameter(f"θ_{i}") for i in range(3)]
    qc.rx(theta[0], 0)
    qc.ry(theta[1], 1)
    qc.cx(0, 1)
    return qc

def build_qcnn_circuit(num_qubits: int) -> QuantumCircuit:
    """
    Assemble the full QCNN ansatz by stacking convolution and pooling layers.
    """
    qc = QuantumCircuit(num_qubits)
    # First convolution
    qc.append(quantum_conv_circuit(int(np.sqrt(num_qubits))), range(num_qubits))
    # First pooling
    qc.append(quantum_pool_circuit(), range(num_qubits))
    # Additional layers could be added here
    return qc

def run_quantum_circuit(circuit: QuantumCircuit, input_vector: np.ndarray) -> float:
    """
    Execute a parameterised quantum circuit with the given input vector
    and return the expectation value of a Z observable on the first qubit.
    """
    # Bind input parameters
    param_binds = {circuit.parameters[i]: input_vector[i] for i in range(len(input_vector))}
    bound_qc = circuit.bind_parameters(param_binds)
    estimator = Estimator()
    observable = SparsePauliOp.from_list([("Z" + "I" * (circuit.num_qubits - 1), 1)])
    result = estimator.run(bound_qc, observables=observable).result()
    expectation = result.values[0]
    return float(expectation)

def get_qcnn_qnn(num_qubits: int) -> EstimatorQNN:
    """
    Return an EstimatorQNN that encapsulates the QCNN ansatz.
    """
    circuit = build_qcnn_circuit(num_qubits)
    estimator = Estimator()
    observable = SparsePauliOp.from_list([("Z" + "I" * (num_qubits - 1), 1)])
    qnn = EstimatorQNN(
        circuit=circuit,
        observables=observable,
        input_params=[],  # No classical feature map here
        weight_params=circuit.parameters,
        estimator=estimator,
    )
    return qnn

__all__ = [
    "quantum_conv_circuit",
    "quantum_pool_circuit",
    "build_qcnn_circuit",
    "run_quantum_circuit",
    "get_qcnn_qnn",
]
