import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals

def QCNNHybrid() -> EstimatorQNN:
    """
    Quantum neural network that mirrors the classical
    ``QCNNHybridModel`` but with a richer ansatz.  The ansatz
    consists of three convolutional layers, three pooling layers,
    and an additional entangling layer that couples all qubits.
    The network uses a parameter‑shift gradient estimator and
    measures the expectation value of ``Z`` on the first qubit.
    """
    algorithm_globals.random_seed = 12345
    estimator = Estimator()

    # ----- Convolutional sub‑circuit -----
    def conv_circuit(params):
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        qc.cx(1, 0)
        qc.rz(np.pi / 2, 0)
        return qc

    # ----- Pooling sub‑circuit -----
    def pool_circuit(params):
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    # ----- Layer helpers -----
    def conv_layer(num_qubits, param_prefix):
        qc = QuantumCircuit(num_qubits, name="ConvLayer")
        qubits = list(range(num_qubits))
        params = ParameterVector(param_prefix, length=num_qubits * 3)
        for pair_index, (q1, q2) in enumerate(zip(qubits[0::2], qubits[1::2])):
            qc.append(conv_circuit(params[pair_index*3:(pair_index+1)*3]), [q1, q2])
            qc.barrier()
        return qc

    def pool_layer(sources, sinks, param_prefix):
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits, name="PoolLayer")
        params = ParameterVector(param_prefix, length=len(sources) * 3)
        for idx, (src, sink) in enumerate(zip(sources, sinks)):
            qc.append(pool_circuit(params[idx*3:(idx+1)*3]), [src, sink])
            qc.barrier()
        return qc

    # ----- Entangling layer -----
    def entangle_layer(num_qubits):
        qc = QuantumCircuit(num_qubits, name="EntangleLayer")
        for i in range(num_qubits - 1):
            qc.cx(i, i + 1)
        return qc

    # ----- Build the full ansatz -----
    ansatz = QuantumCircuit(8, name="Ansatz")

    # Convolution and pooling sequence
    ansatz.compose(conv_layer(8, "c1"), inplace=True)
    ansatz.compose(pool_layer([0,1,2,3], [4,5,6,7], "p1"), inplace=True)

    ansatz.compose(conv_layer(4, "c2"), inplace=True)
    ansatz.compose(pool_layer([0,1], [2,3], "p2"), inplace=True)

    ansatz.compose(conv_layer(2, "c3"), inplace=True)
    ansatz.compose(pool_layer([0], [1], "p3"), inplace=True)

    # Additional entangling layer to increase expressivity
    ansatz.compose(entangle_layer(8), inplace=True)

    # Feature map
    feature_map = ZFeatureMap(8)

    # Combine feature map and ansatz
    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, inplace=True)
    circuit.compose(ansatz, inplace=True)

    # Observable: Z on first qubit
    observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

    # Build the EstimatorQNN
    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )
    return qnn

__all__ = ["QCNNHybrid"]
