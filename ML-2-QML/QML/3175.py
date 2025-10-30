import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

def _conv_circuit(params):
    """Two‑qubit convolution subcircuit used in the QCNN ansatz."""
    qc = QuantumCircuit(2)
    qc.rz(-qiskit.math.pi/2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    qc.cx(1, 0)
    qc.rz(qiskit.math.pi/2, 0)
    return qc

def _pool_circuit(params):
    """Two‑qubit pooling subcircuit used in the QCNN ansatz."""
    qc = QuantumCircuit(2)
    qc.rz(-qiskit.math.pi/2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc

def _conv_layer(num_qubits, prefix):
    """Build a convolution layer that operates on adjacent qubit pairs."""
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(prefix, length=(num_qubits//2) * 3)
    idx = 0
    for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
        sub = _conv_circuit(params[idx:idx+3])
        qc.append(sub, [q1, q2])
        idx += 3
    return qc

def _pool_layer(num_qubits, prefix):
    """Build a pooling layer that operates on adjacent qubit pairs."""
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(prefix, length=(num_qubits//2) * 3)
    idx = 0
    for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
        sub = _pool_circuit(params[idx:idx+3])
        qc.append(sub, [q1, q2])
        idx += 3
    return qc

def build_qcnn_ansatz(num_qubits: int = 8,
                      conv_depth: int = 3,
                      pool_depth: int = 3) -> QuantumCircuit:
    """
    Construct the full QCNN ansatz circuit.
    The circuit alternates convolution and pooling layers,
    halving the number of qubits after each pooling step.
    """
    qc = QuantumCircuit(num_qubits)
    current_qubits = num_qubits
    for d in range(conv_depth):
        qc.append(_conv_layer(current_qubits, f"c{d+1}"), range(current_qubits))
        qc.append(_pool_layer(current_qubits, f"p{d+1}"), range(current_qubits))
        current_qubits //= 2  # qubits are reduced by pooling
    return qc

def build_qcnn_qnn(feature_dim: int = 8) -> EstimatorQNN:
    """
    Build a QCNN EstimatorQNN that can be used as a quantum head
    in a hybrid model.  The returned object is callable and
    evaluates the expectation value of a single Z observable.
    """
    # Feature map
    feature_map = ZFeatureMap(feature_dim)

    # Ansatz
    ansatz = build_qcnn_ansatz(feature_dim)

    # Observable: single‑qubit Z on the first qubit
    observable = SparsePauliOp.from_list([("Z" + "I" * (feature_dim - 1), 1)])

    # Estimator
    estimator = Estimator()

    # Build EstimatorQNN
    qnn = EstimatorQNN(
        circuit=ansatz.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )
    return qnn

__all__ = ["build_qcnn_ansatz", "build_qcnn_qnn"]
