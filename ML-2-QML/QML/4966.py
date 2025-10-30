import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector, Parameter
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit_machine_learning.neural_networks import EstimatorQNN, SamplerQNN

def _conv_circuit(params):
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

def _pool_circuit(params):
    qc = QuantumCircuit(2)
    qc.rz(-np.pi/2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc

def _conv_layer(num_qubits, name, prefix):
    qc = QuantumCircuit(num_qubits, name=name)
    params = ParameterVector(prefix, length=num_qubits * 3)
    for i in range(0, num_qubits, 2):
        idx = (i // 2) * 3
        qc.compose(_conv_circuit(params[idx:idx+3]), [i, i+1], inplace=True)
    return qc

def _pool_layer(sources, sinks, name, prefix):
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name=name)
    params = ParameterVector(prefix, length=(num_qubits // 2) * 3)
    for idx, (s, t) in enumerate(zip(sources, sinks)):
        q_params = params[idx*3:idx*3+3]
        qc.compose(_pool_circuit(q_params), [s, t], inplace=True)
    return qc

def QCNNQuantum():
    """Build the full QCNN variational circuit and wrap it as an EstimatorQNN."""
    feature_map = ZFeatureMap(8)
    ansatz = QuantumCircuit(8, name="Ansatz")

    ansatz.compose(_conv_layer(8, "conv1", "c1"), inplace=True)
    ansatz.compose(_pool_layer([0,1,2,3], [4,5,6,7], "pool1", "p1"), inplace=True)
    ansatz.compose(_conv_layer(4, "conv2", "c2"), inplace=True)
    ansatz.compose(_pool_layer([0,1], [2,3], "pool2", "p2"), inplace=True)
    ansatz.compose(_conv_layer(2, "conv3", "c3"), inplace=True)
    ansatz.compose(_pool_layer([0], [1], "pool3", "p3"), inplace=True)

    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, inplace=True)
    circuit.compose(ansatz, inplace=True)

    observable = SparsePauliOp.from_list([("Z" + "I"*7, 1)])
    estimator = StatevectorEstimator()
    qnn = EstimatorQNN(
        circuit=circuit,
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator
    )
    return qnn

class HybridQNN:
    """Encapsulates both EstimatorQNN (regression) and SamplerQNN (sampling)."""
    def __init__(self):
        self.estimator = QCNNQuantum()

        # Sampler circuit
        inputs = ParameterVector("input", 2)
        weights = ParameterVector("weight", 4)
        qc = QuantumCircuit(2)
        qc.ry(inputs[0], 0)
        qc.ry(inputs[1], 1)
        qc.cx(0, 1)
        qc.ry(weights[0], 0)
        qc.ry(weights[1], 1)
        qc.cx(0, 1)
        qc.ry(weights[2], 0)
        qc.ry(weights[3], 1)

        sampler = StatevectorSampler()
        self.sampler = SamplerQNN(
            circuit=qc,
            input_params=inputs,
            weight_params=weights,
            sampler=sampler
        )

    def predict(self, x):
        """Return the expectation value from the EstimatorQNN."""
        return self.estimator.predict(x)

    def sample(self, x):
        """Return sample counts from the SamplerQNN."""
        return self.sampler.sample(x)

__all__ = ["QCNNQuantum", "HybridQNN"]
