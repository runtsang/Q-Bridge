"""Hybrid quantum estimator that chains a variational regression, a quantum convolution filter
and a quantum self‑attention block."""
from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
from qiskit.circuit.random import random_circuit
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import Estimator as QiskitEstimator
from qiskit.quantum_info import SparsePauliOp

# Quantum convolution
class QuanvCircuit:
    def __init__(self, kernel_size: int, backend, shots: int, threshold: float):
        self.n_qubits = kernel_size ** 2
        self.backend = backend
        self.shots = shots
        self.threshold = threshold
        self.circuit = QuantumCircuit(self.n_qubits)
        self.theta = [Parameter(f'theta{i}') for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self.circuit.rx(self.theta[i], i)
        self.circuit.barrier()
        self.circuit += random_circuit(self.n_qubits, 2)
        self.circuit.measure_all()

    def run(self, data: np.ndarray) -> float:
        data = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data:
            bind = {self.theta[i]: np.pi if val > self.threshold else 0 for i, val in enumerate(dat)}
            param_binds.append(bind)
        job = qiskit.execute(self.circuit, self.backend, shots=self.shots, parameter_binds=param_binds)
        result = job.result().get_counts(self.circuit)
        counts = 0
        for key, val in result.items():
            ones = sum(int(b) for b in key)
            counts += ones * val
        return counts / (self.shots * self.n_qubits)

# Quantum self‑attention
class QuantumSelfAttention:
    def __init__(self, n_qubits: int, backend):
        self.n_qubits = n_qubits
        self.backend = backend
        self.qr = QuantumRegister(n_qubits, 'q')
        self.cr = ClassicalRegister(n_qubits, 'c')

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

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, shots: int = 1024):
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = qiskit.execute(circuit, self.backend, shots=shots)
        return job.result().get_counts(circuit)

# Hybrid estimator
class EstimatorQNN:
    """Composite quantum estimator that chains a variational regression,
    a quantum convolution filter and a quantum self‑attention block."""
    def __init__(self,
                 backend=None,
                 shots: int = 1024,
                 conv_threshold: float = 127.0,
                 attention_n_qubits: int = 4):
        if backend is None:
            backend = qiskit.Aer.get_backend('qasm_simulator')
        self.backend = backend
        self.shots = shots
        # regression circuit using QiskitML EstimatorQNN
        input_params = [Parameter('x0'), Parameter('x1')]
        weight_params = [Parameter('w0'), Parameter('w1'), Parameter('w2')]
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.ry(input_params[0], 0)
        qc.rx(weight_params[0], 0)
        qc.rx(weight_params[1], 0)
        qc.ry(weight_params[2], 0)
        observable = SparsePauliOp.from_list([('Y', 1)])
        estimator = QiskitEstimator()
        self.regressor = QiskitEstimatorQNN(circuit=qc,
                                            observables=observable,
                                            input_params=input_params,
                                            weight_params=weight_params,
                                            estimator=estimator)
        # quantum convolution
        self.conv = QuanvCircuit(kernel_size=2, backend=self.backend,
                                 shots=shots, threshold=conv_threshold)
        # quantum attention
        self.attention = QuantumSelfAttention(n_qubits=attention_n_qubits,
                                              backend=self.backend)

    def run(self, data: np.ndarray) -> float:
        """Run the full hybrid pipeline on a 2‑D input array.

        Parameters
        ----------
        data
            2‑D array of shape (2, 2) representing a single pixel patch.
        """
        # 1. quantum convolution
        conv_out = self.conv.run(data)
        # 2. quantum attention
        rot = np.random.rand(3 * self.attention.n_qubits)
        ent = np.random.rand(self.attention.n_qubits - 1)
        attn_counts = self.attention.run(rot, ent)
        # 3. regression
        reg_input = np.array([conv_out,
                              attn_counts.get('1' * self.attention.n_qubits, 0) / self.shots])
        reg_output = self.regressor.predict(reg_input)
        return reg_output[0]

__all__ = ["EstimatorQNN"]
