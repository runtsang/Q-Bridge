import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.circuit import Parameter
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp

class SelfAttentionHybridQML:
    """Quantum self‑attention block with an embedded quantum neural network estimator."""
    def __init__(self, n_qubits: int = 4):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")
        self.backend = Aer.get_backend("qasm_simulator")

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        qc = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            qc.rx(rotation_params[3 * i], i)
            qc.ry(rotation_params[3 * i + 1], i)
            qc.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            qc.crx(entangle_params[i], i, i + 1)
        return qc

    def run_attention(self, rotation_params: np.ndarray, entangle_params: np.ndarray, shots: int = 1024):
        qc = self._build_circuit(rotation_params, entangle_params)
        qc.measure(self.qr, self.cr)
        job = execute(qc, self.backend, shots=shots)
        return job.result().get_counts(qc)

    def build_qnn(self, input_params: list[Parameter], weight_params: list[Parameter]) -> EstimatorQNN:
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.ry(input_params[0], 0)
        qc.rx(weight_params[0], 0)
        observable = SparsePauliOp.from_list([("Y" * qc.num_qubits, 1)])
        estimator = StatevectorEstimator()
        return EstimatorQNN(
            circuit=qc,
            observables=observable,
            input_params=[input_params[0]],
            weight_params=[weight_params[0]],
            estimator=estimator
        )

    def run_qnn(self, estimator_qnn: EstimatorQNN, inputs: np.ndarray):
        return estimator_qnn.predict(inputs)

__all__ = ["SelfAttentionHybridQML"]
