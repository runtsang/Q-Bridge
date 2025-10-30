import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator as StatevectorEstimator
from qiskit.circuit.library import ZFeatureMap
from qiskit_machine_learning.neural_networks import EstimatorQNN

def HybridQCNN(num_qubits: int = 8, depth: int = 3):
    """Return a QCNN EstimatorQNN combining convolutional and pooling layers."""
    feature_map = ZFeatureMap(num_qubits)

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

    def conv_layer(num_qubits, param_prefix):
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(param_prefix, length=num_qubits * 3 // 2)
        idx = 0
        for q1, q2 in zip(range(0, num_qubits, 2), range(1, num_qubits, 2)):
            sub = conv_circuit(params[idx:idx + 3])
            qc.append(sub, [q1, q2])
            idx += 3
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

    def pool_layer(num_qubits, param_prefix):
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
        idx = 0
        for q1, q2 in zip(range(num_qubits // 2), range(num_qubits // 2, num_qubits)):
            sub = pool_circuit(params[idx:idx + 3])
            qc.append(sub, [q1, q2])
            idx += 3
        return qc

    ansatz = QuantumCircuit(num_qubits)
    ansatz.compose(conv_layer(num_qubits, "c1"), inplace=True)
    ansatz.compose(pool_layer(num_qubits, "p1"), inplace=True)
    for d in range(2, depth + 1):
        ansatz.compose(conv_layer(num_qubits, f"c{d}"), inplace=True)
        ansatz.compose(pool_layer(num_qubits, f"p{d}"), inplace=True)

    observable = SparsePauliOp.from_list([("Z" + "I" * (num_qubits - 1), 1)])
    estimator = StatevectorEstimator()
    qnn = EstimatorQNN(
        circuit=ansatz.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )
    return qnn

def QuantumConvFilter(kernel_size=2, backend='qasm_simulator', shots=100, threshold=127):
    """Quantum filter that emulates a classical convolution via a random circuit."""
    from qiskit import Aer, execute
    from qiskit.circuit.random import random_circuit

    class QuanvCircuit:
        def __init__(self, kernel_size, backend, shots, threshold):
            self.n_qubits = kernel_size ** 2
            self._circuit = QuantumCircuit(self.n_qubits)
            self.theta = [ParameterVector(f"theta{i}") for i in range(self.n_qubits)]
            for i in range(self.n_qubits):
                self._circuit.rx(self.theta[i], i)
            self._circuit.barrier()
            self._circuit += random_circuit(self.n_qubits, 2)
            self._circuit.measure_all()
            self.backend = backend
            self.shots = shots
            self.threshold = threshold

        def run(self, data):
            data = np.reshape(data, (1, self.n_qubits))
            param_binds = []
            for dat in data:
                bind = {}
                for i, val in enumerate(dat):
                    bind[self.theta[i]] = np.pi if val > self.threshold else 0
                param_binds.append(bind)
            job = execute(self._circuit, self.backend, shots=self.shots, parameter_binds=param_binds)
            result = job.result().get_counts(self._circuit)
            counts = 0
            for key, val in result.items():
                ones = sum(int(bit) for bit in key)
                counts += ones * val
            return counts / (self.shots * self.n_qubits)

    backend = Aer.get_backend(backend)
    return QuanvCircuit(kernel_size, backend, shots, threshold)

def build_classifier_circuit(num_qubits: int, depth: int):
    """Constructs a data‑uploading classifier circuit."""
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
    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return circuit, list(encoding), list(weights), observables
