from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
import numpy as np

def conv_circuit(params):
    """Two‑qubit convolutional unitary."""
    target = QuantumCircuit(2)
    target.rz(-np.pi / 2, 1)
    target.cx(1, 0)
    target.rz(params[0], 0)
    target.ry(params[1], 1)
    target.cx(0, 1)
    target.ry(params[2], 1)
    target.cx(1, 0)
    target.rz(np.pi / 2, 0)
    return target

def pool_circuit(params):
    """Two‑qubit pooling unitary."""
    target = QuantumCircuit(2)
    target.rz(-np.pi / 2, 1)
    target.cx(1, 0)
    target.rz(params[0], 0)
    target.ry(params[1], 1)
    target.cx(0, 1)
    target.ry(params[2], 1)
    return target

def conv_layer(num_qubits, param_prefix):
    """Construct a convolutional layer over paired qubits."""
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc = qc.compose(conv_circuit(params[param_index : param_index + 3]), [q1, q2])
        qc.barrier()
        param_index += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        qc = qc.compose(conv_circuit(params[param_index : param_index + 3]), [q1, q2])
        qc.barrier()
        param_index += 3
    qc_inst = qc.to_instruction()
    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, qubits)
    return qc

def pool_layer(sources, sinks, param_prefix):
    """Construct a pooling layer that maps source qubits to sink qubits."""
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for source, sink in zip(sources, sinks):
        qc = qc.compose(pool_circuit(params[param_index : param_index + 3]), [source, sink])
        qc.barrier()
        param_index += 3
    qc_inst = qc.to_instruction()
    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, range(num_qubits))
    return qc

def simple_1q_circuit(params):
    """Simple one‑qubit rotation block used as a final variational layer."""
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.ry(params[0], 0)
    qc.rx(params[1], 0)
    return qc

def build_ansatz():
    """Assemble the full QCNN ansatz with an additional one‑qubit layer."""
    feature_map = ZFeatureMap(8)
    ansatz = QuantumCircuit(8, name="Ansatz")

    # First convolutional and pooling stages
    ansatz.compose(conv_layer(8, "c1"), list(range(8)), inplace=True)
    ansatz.compose(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), list(range(8)), inplace=True)

    # Second convolutional and pooling stages
    ansatz.compose(conv_layer(4, "c2"), list(range(4, 8)), inplace=True)
    ansatz.compose(pool_layer([0, 1], [2, 3], "p2"), list(range(4, 8)), inplace=True)

    # Third convolutional and pooling stages
    ansatz.compose(conv_layer(2, "c3"), list(range(6, 8)), inplace=True)
    ansatz.compose(pool_layer([0], [1], "p3"), list(range(6, 8)), inplace=True)

    # Final one‑qubit variational block
    simple_params = ParameterVector("s", length=2)
    ansatz.compose(simple_1q_circuit(simple_params), [0], inplace=True)

    return ansatz, feature_map

class HybridEstimatorQNN:
    """Quantum neural network combining QCNN ansatz with a simple one‑qubit rotation."""
    def __init__(self):
        estimator = StatevectorEstimator()
        ansatz, feature_map = build_ansatz()
        observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])
        self.estimator_qnn = EstimatorQNN(
            circuit=ansatz,
            observables=observable,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            estimator=estimator,
        )

    def __call__(self, inputs: dict) -> np.ndarray:
        """Evaluate the QNN given a mapping of input parameters to numerical values."""
        return self.estimator_qnn(inputs)

    @property
    def input_params(self):
        """Return the list of input parameters for the feature map."""
        return self.estimator_qnn.input_params
