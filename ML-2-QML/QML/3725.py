import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector, Parameter
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit.utils import algorithm_globals
from qiskit_machine_learning.neural_networks import EstimatorQNN

def UnifiedQCNN(num_qubits: int = 8,
                feature_dim: int = 8,
                seed: int | None = 12345):
    """
    Builds a hybrid QCNN ansatz that combines convolution‑pooling layers
    from the QCNN seed with a simple 1‑qubit EstimatorQNN read‑out.
    Returns a qiskit_machine_learning.neural_networks.EstimatorQNN instance.
    """
    algorithm_globals.random_seed = seed
    estimator = Estimator()

    # Feature map
    feature_map = ZFeatureMap(feature_dim)

    # Helper circuits
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

    def pool_circuit(params):
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    # Convolutional layer
    def conv_layer(num_qubits, param_prefix):
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(param_prefix, length=num_qubits * 3)
        for i in range(0, num_qubits, 2):
            sub = conv_circuit(params[i*3:(i+1)*3])
            qc.append(sub, [i, i+1])
        return qc

    # Pooling layer
    def pool_layer(sources, sinks, param_prefix):
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(param_prefix, length=len(sources) * 3)
        for src, snk, idx in zip(sources, sinks, range(len(sources))):
            sub = pool_circuit(params[idx*3:(idx+1)*3])
            qc.append(sub, [src, snk])
        return qc

    # Simple 1‑qubit read‑out (EstimatorQNN style)
    weight_param = Parameter("w")
    simple_qc = QuantumCircuit(1)
    simple_qc.h(0)
    simple_qc.ry(weight_param, 0)

    # Build the full ansatz
    ansatz = QuantumCircuit(num_qubits)

    # First convolutional layer
    ansatz.append(conv_layer(num_qubits, "c1"), range(num_qubits))

    # First pooling layer
    ansatz.append(pool_layer(list(range(num_qubits//2)),
                             list(range(num_qubits//2, num_qubits)),
                             "p1"), range(num_qubits))

    # Second convolutional layer
    ansatz.append(conv_layer(num_qubits//2, "c2"), range(num_qubits//2))

    # Second pooling layer
    ansatz.append(pool_layer(list(range(num_qubits//4)),
                             list(range(num_qubits//4, num_qubits//2)),
                             "p2"), range(num_qubits//2))

    # Third convolutional layer
    ansatz.append(conv_layer(num_qubits//4, "c3"), range(num_qubits//4))

    # Third pooling layer
    ansatz.append(pool_layer([0], [1], "p3"), range(num_qubits//4))

    # Append simple read‑out to the first qubit
    ansatz.append(simple_qc, [0])

    # Observable and EstimatorQNN
    observable = SparsePauliOp.from_list([("Z" + "I" * (num_qubits-1), 1)])
    weight_params = ansatz.parameters + [weight_param]

    qnn = EstimatorQNN(
        circuit=ansatz.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=weight_params,
        estimator=estimator,
    )
    return qnn
