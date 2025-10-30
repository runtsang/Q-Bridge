import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.circuit import Parameter, ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import Estimator as StatevectorEstimator

class HybridSelfAttention:
    """Quantum hybrid model that bundles quantum self‑attention, quantum LSTM, QCNN and QNN sub‑modules."""

    def __init__(self, n_qubits: int = 4):
        self.n_qubits = n_qubits
        self.backend = Aer.get_backend('qasm_simulator')
        self.self_attention_circuit = self._build_self_attention()
        self.estimator = self._build_estimator()
        self.qcnn_circuit = self._build_qcnn()
        self.qlstm_circuit = self._build_qlstm()

    # --------------------- Self‑attention ---------------------
    def _build_self_attention(self) -> QuantumCircuit:
        qr = QuantumRegister(self.n_qubits, 'q')
        cr = ClassicalRegister(self.n_qubits, 'c')
        circuit = QuantumCircuit(qr, cr)
        for i in range(self.n_qubits):
            circuit.rx(np.pi / 4, i)
            circuit.ry(np.pi / 4, i)
            circuit.rz(np.pi / 4, i)
        for i in range(self.n_qubits - 1):
            circuit.cx(i, i + 1)
        circuit.measure(qr, cr)
        return circuit

    def run_self_attention(self, shots: int = 1024):
        job = execute(self.self_attention_circuit, self.backend, shots=shots)
        return job.result().get_counts(self.self_attention_circuit)

    # --------------------- Estimator QNN ---------------------
    def _build_estimator(self) -> EstimatorQNN:
        theta = Parameter('θ')
        w = Parameter('w')
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.ry(theta, 0)
        qc.rx(w, 0)
        observable = SparsePauliOp.from_list([('Y', 1)])
        estimator = StatevectorEstimator()
        return EstimatorQNN(circuit=qc,
                            observables=observable,
                            input_params=[theta],
                            weight_params=[w],
                            estimator=estimator)

    # --------------------- QCNN ---------------------
    def _build_qcnn(self) -> QuantumCircuit:
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

        def conv_layer(num_qubits, prefix):
            qc = QuantumCircuit(num_qubits)
            qubits = list(range(num_qubits))
            param_vec = ParameterVector(prefix, length=num_qubits * 3)
            for i in range(0, num_qubits, 2):
                qc.append(conv_circuit(param_vec[i*3:(i+2)*3]), [qubits[i], qubits[i+1]])
            return qc

        def pool_layer(sources, sinks, prefix):
            num = len(sources) + len(sinks)
            qc = QuantumCircuit(num)
            param_vec = ParameterVector(prefix, length=num//2 * 3)
            for i, (s, t) in enumerate(zip(sources, sinks)):
                qc.append(pool_circuit(param_vec[i*3:(i+1)*3]), [s, t])
            return qc

        # Build a 8‑qubit QCNN ansatz as in the reference
        ansatz = QuantumCircuit(8)
        ansatz.compose(conv_layer(8, 'c1'), inplace=True)
        ansatz.compose(pool_layer([0,1,2,3], [4,5,6,7], 'p1'), inplace=True)
        ansatz.compose(conv_layer(4, 'c2'), inplace=True)
        ansatz.compose(pool_layer([0,1], [2,3], 'p2'), inplace=True)
        ansatz.compose(conv_layer(2, 'c3'), inplace=True)
        ansatz.compose(pool_layer([0], [1], 'p3'), inplace=True)
        return ansatz

    # --------------------- Quantum LSTM ---------------------
    def _build_qlstm(self) -> QuantumCircuit:
        # A minimal quantum LSTM cell: encode input into qubits, apply rotations,
        # entangle with CNOTs, and measure.  This is a toy illustration.
        qc = QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
            qc.h(i)
            qc.rx(np.pi / 2, i)
        for i in range(self.n_qubits - 1):
            qc.cx(i, i + 1)
        qc.measure_all()
        return qc

    def run_qlstm(self, shots: int = 1024):
        job = execute(self.qlstm_circuit, self.backend, shots=shots)
        return job.result().get_counts(self.qlstm_circuit)
