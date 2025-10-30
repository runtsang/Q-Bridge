import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.circuit.random import random_circuit
from qiskit.circuit.library import ZFeatureMap
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit.quantum_info import SparsePauliOp

# ────────────────────── Quantum convolution filter  ──────────────────────
def Conv():
    class QuanvCircuit:
        def __init__(self, kernel_size, backend, shots, threshold):
            self.n_qubits = kernel_size ** 2
            self._circuit = qiskit.QuantumCircuit(self.n_qubits)
            self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
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
            job = execute(self._circuit, self.backend,
                          shots=self.shots, parameter_binds=param_binds)
            result = job.result().get_counts(self._circuit)
            counts = 0
            for key, val in result.items():
                ones = sum(int(bit) for bit in key)
                counts += ones * val
            return counts / (self.shots * self.n_qubits)

    backend = Aer.get_backend("qasm_simulator")
    circuit = QuanvCircuit(filter_size=2, backend=backend, shots=100, threshold=127)
    return circuit

# ────────────────────── Quantum self‑attention  ──────────────────────
def SelfAttention():
    class QuantumSelfAttention:
        def __init__(self, n_qubits: int):
            self.n_qubits = n_qubits
            self.qr = QuantumRegister(n_qubits, "q")
            self.cr = ClassicalRegister(n_qubits, "c")

        def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
            circuit = QuantumCircuit(self.qr, self.cr)
            for i in range(self.n_qubits):
                circuit.rx(rotation_params[3 * i], i)
                circuit.ry(rotation_params[3 * i + 1], i)
                circuit.rz(rotation_params[3 * i + 2], i)
            for i in range(self.n_qubits - 1):
                circuit.crx(entangle_params[i], i, i + 1)
            circuit.measure(self.qr, self.cr)
            return circuit

        def run(self, backend, rotation_params: np.ndarray, entangle_params: np.ndarray, shots: int = 1024):
            circuit = self._build_circuit(rotation_params, entangle_params)
            job = execute(circuit, backend, shots=shots)
            return job.result().get_counts(circuit)

    backend = Aer.get_backend("qasm_simulator")
    return QuantumSelfAttention(n_qubits=4)

# ────────────────────── QCNN Ansatz (Qiskit)  ──────────────────────
def QCNN():
    algorithm_globals.random_seed = 12345
    estimator = Estimator()

    def conv_circuit(params):
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

    def conv_layer(num_qubits, param_prefix):
        qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
        qubits = list(range(num_qubits))
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits * 3)
        for q1, q2 in zip(qubits[0::2], qubits[1::2]):
            qc = qc.compose(conv_circuit(params[param_index : (param_index + 3)]), [q1, q2])
            qc.barrier()
            param_index += 3
        for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
            qc = qc.compose(conv_circuit(params[param_index : (param_index + 3)]), [q1, q2])
            qc.barrier()
            param_index += 3

        qc_inst = qc.to_instruction()
        qc = QuantumCircuit(num_qubits)
        qc.append(qc_inst, qubits)
        return qc

    def pool_circuit(params):
        target = QuantumCircuit(2)
        target.rz(-np.pi / 2, 1)
        target.cx(1, 0)
        target.rz(params[0], 0)
        target.ry(params[1], 1)
        target.cx(0, 1)
        target.ry(params[2], 1)
        return target

    def pool_layer(sources, sinks, param_prefix):
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits, name="Pooling Layer")
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
        for source, sink in zip(sources, sinks):
            qc = qc.compose(pool_circuit(params[param_index : (param_index + 3)]), [source, sink])
            qc.barrier()
            param_index += 3

        qc_inst = qc.to_instruction()
        qc = QuantumCircuit(num_qubits)
        qc.append(qc_inst, range(num_qubits))
        return qc

    # Build full QCNN ansatz
    ansatz = QuantumCircuit(8, name="Ansatz")
    ansatz.compose(conv_layer(8, "c1"), list(range(8)), inplace=True)
    ansatz.compose(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), list(range(8)), inplace=True)
    ansatz.compose(conv_layer(4, "c2"), list(range(4, 8)), inplace=True)
    ansatz.compose(pool_layer([0, 1], [2, 3], "p2"), list(range(4, 8)), inplace=True)
    ansatz.compose(conv_layer(2, "c3"), list(range(6, 8)), inplace=True)
    ansatz.compose(pool_layer([0], [1], "p3"), list(range(6, 8)), inplace=True)

    feature_map = ZFeatureMap(8)
    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, range(8), inplace=True)
    circuit.compose(ansatz, range(8), inplace=True)

    observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])
    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )
    return qnn

# ────────────────────── Hybrid quantum wrapper  ──────────────────────
class FraudDetectionQuantumHybrid:
    """Quantum circuit that mirrors the classical FraudDetectionHybrid architecture."""

    def __init__(self):
        # Feature map and QCNN ansatz
        self.qcnn = QCNN()
        # Quantum self‑attention sub‑circuit
        self.attention = SelfAttention()
        # Quantum convolution filter
        self.conv = Conv()
        # Build composite circuit
        self.circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        # Start with 8‑qubit feature map + QCNN ansatz
        base_circuit = self.qcnn.circuit
        # Append attention sub‑circuit (reuse 4 qubits for illustration)
        attention_circ = QuantumCircuit(4, name="Attention")
        # Dummy parameters – in practice these would be trainable
        rot = np.random.rand(12)
        ent = np.random.rand(3)
        attention_circ = self.attention._build_circuit(rot, ent)
        # Embed attention as a sub‑instruction into the 8‑qubit circuit
        attention_inst = attention_circ.to_instruction()
        base_circuit.append(attention_inst, range(4))
        # Append convolution filter (acts on 2×2 block → 4 qubits)
        conv_inst = self.conv._circuit.to_instruction()
        base_circuit.append(conv_inst, range(4))
        return base_circuit

    def evaluate(self, input_vector: np.ndarray, shots: int = 1024) -> float:
        """Run the composite circuit on a backend and return expectation value."""
        backend = Aer.get_backend("qasm_simulator")
        job = execute(self.circuit, backend, shots=shots)
        result = job.result().get_counts(self.circuit)
        # Compute average probability of measuring |1> on first qubit as a proxy
        total = 0
        for bitstring, count in result.items():
            total += int(bitstring[-1]) * count
        return total / (shots * len(self.circuit.qubits))

__all__ = ["FraudDetectionQuantumHybrid", "QCNN", "SelfAttention", "Conv"]
