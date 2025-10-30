import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals

# ------------------------------------------------------------------
# Convolution and pooling primitives (quantum QCNN blocks)
# ------------------------------------------------------------------
def conv_circuit(params):
    qc = QuantumCircuit(2)
    qc.rz(-np.pi/2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    qc.cx(1, 0)
    qc.rz(np.pi/2, 0)
    return qc

def conv_layer(num_qubits, param_prefix):
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * 3 // 2)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc.compose(conv_circuit(params[param_index:param_index+3]),
                   [q1, q2], inplace=True)
        qc.barrier()
        param_index += 3
    return qc

def pool_circuit(params):
    qc = QuantumCircuit(2)
    qc.rz(-np.pi/2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc

def pool_layer(sources, sinks, param_prefix):
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for source, sink in zip(sources, sinks):
        qc.compose(pool_circuit(params[param_index:param_index+3]),
                   [source, sink], inplace=True)
        qc.barrier()
        param_index += 3
    return qc

# ------------------------------------------------------------------
# Incremental data‑uploading classifier ansatz
# ------------------------------------------------------------------
def build_classifier_circuit(num_qubits: int,
                             depth: int) -> Tuple[QuantumCircuit,
                                                  Iterable,
                                                  Iterable,
                                                  list[SparsePauliOp]]:
    """
    Construct a simple layered ansatz with explicit encoding and variational
    parameters, mirroring the classical build_classifier_circuit.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    observables = [SparsePauliOp("I"*i + "Z" + "I"*(num_qubits-i-1))
                   for i in range(num_qubits)]
    return circuit, list(encoding), list(weights), observables

# ------------------------------------------------------------------
# Hybrid QCNN with classifier
# ------------------------------------------------------------------
def HybridQCNN(num_qubits: int = 8,
               conv_depth: int = 3,
               classifier_depth: int = 2) -> EstimatorQNN:
    """
    Assemble a QCNN ansatz that includes convolution/pooling blocks
    followed by a variational classifier.  Returns an EstimatorQNN
    ready for hybrid training.
    """
    algorithm_globals.random_seed = 12345
    estimator = Estimator()

    # Feature map
    feature_map = ZFeatureMap(num_qubits)

    # Build the ansatz
    ansatz = QuantumCircuit(num_qubits)

    # Convolution / pooling layers
    for i in range(conv_depth):
        ansatz.compose(conv_layer(num_qubits, f"c{i+1}"),
                       range(num_qubits), inplace=True)
        ansatz.compose(pool_layer(list(range(num_qubits)),
                                  list(range(num_qubits)),
                                  f"p{i+1}"),
                       range(num_qubits), inplace=True)

    # Append the classifier ansatz
    clf_circuit, _, _, _ = build_classifier_circuit(num_qubits, classifier_depth)
    ansatz.compose(clf_circuit, range(num_qubits), inplace=True)

    # Full circuit: feature map + ansatz
    circuit = QuantumCircuit(num_qubits)
    circuit.compose(feature_map, range(num_qubits), inplace=True)
    circuit.compose(ansatz, range(num_qubits), inplace=True)

    # Observables: single‑qubit Z on the first qubit
    observable = SparsePauliOp.from_list([("Z" + "I" * (num_qubits - 1), 1)])

    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator
    )
    return qnn

__all__ = ["HybridQCNN", "build_classifier_circuit", "conv_layer", "pool_layer"]
