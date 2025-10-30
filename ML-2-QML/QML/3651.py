from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit_machine_learning.neural_networks import EstimatorQNN, SamplerQNN
from qiskit_machine_learning.utils import algorithm_globals

def HybridQCNN() -> tuple[EstimatorQNN, SamplerQNN]:
    """
    Returns a tuple containing:
    1. An EstimatorQNN that implements the QCNN classification circuit.
    2. A SamplerQNN that implements a simple 2‑qubit sampler circuit.
    """
    algorithm_globals.random_seed = 12345
    estimator = StatevectorEstimator()
    sampler = StatevectorSampler()

    # ----- Convolution block -----
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

    def conv_layer(num_qubits, prefix):
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(prefix, length=num_qubits * 3)
        for i in range(0, num_qubits, 2):
            qc.append(conv_circuit(params[i:i+3]), [i, i+1])
            qc.barrier()
        return qc

    # ----- Pooling block -----
    def pool_circuit(params):
        qc = QuantumCircuit(2)
        qc.rz(-np.pi/2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    def pool_layer(sources, sinks, prefix):
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(prefix, length=(num_qubits // 2) * 3)
        for src, snk in zip(sources, sinks):
            qc.append(pool_circuit(params[0:3]), [src, snk])
            qc.barrier()
            params = params[3:]
        return qc

    # ----- Feature map -----
    feature_map = ZFeatureMap(8)

    # ----- Classification ansatz -----
    class_ansatz = QuantumCircuit(8)
    class_ansatz.compose(conv_layer(8, 'c1'), list(range(8)), inplace=True)
    class_ansatz.compose(pool_layer([0,1,2,3], [4,5,6,7], 'p1'), list(range(8)), inplace=True)
    class_ansatz.compose(conv_layer(4, 'c2'), list(range(4,8)), inplace=True)
    class_ansatz.compose(pool_layer([0,1], [2,3], 'p2'), list(range(4,8)), inplace=True)
    class_ansatz.compose(conv_layer(2, 'c3'), list(range(6,8)), inplace=True)
    class_ansatz.compose(pool_layer([0], [1], 'p3'), list(range(6,8)), inplace=True)

    # ----- Combine feature map and ansatz -----
    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, range(8), inplace=True)
    circuit.compose(class_ansatz, range(8), inplace=True)

    observable = SparsePauliOp.from_list([('Z' + 'I'*7, 1)])

    classification_qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=class_ansatz.parameters,
        estimator=estimator
    )

    # ----- Sampler ansatz -----
    input_params = ParameterVector('input', 2)
    weight_params = ParameterVector('weight', 4)
    sampler_circuit = QuantumCircuit(2)
    sampler_circuit.ry(input_params[0], 0)
    sampler_circuit.ry(input_params[1], 1)
    sampler_circuit.cx(0, 1)
    sampler_circuit.ry(weight_params[0], 0)
    sampler_circuit.ry(weight_params[1], 1)
    sampler_circuit.cx(0, 1)
    sampler_circuit.ry(weight_params[2], 0)
    sampler_circuit.ry(weight_params[3], 1)

    sampler_qnn = SamplerQNN(
        circuit=sampler_circuit,
        input_params=input_params,
        weight_params=weight_params,
        sampler=sampler
    )

    return classification_qnn, sampler_qnn
