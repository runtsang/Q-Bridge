"""Hybrid quantum classifier integrating data encoding, variational ansatz, optional self‑attention and FCL emulation."""
from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


class QuantumSelfAttention:
    """Quantum self‑attention block using rotation and crx entanglement."""
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        qc = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            qc.rx(rotation_params[3 * i], i)
            qc.ry(rotation_params[3 * i + 1], i)
            qc.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            qc.crx(entangle_params[i], i, i + 1)
        qc.measure(self.qr, self.cr)
        return qc

    def run(self, backend, rotation_params: np.ndarray, entangle_params: np.ndarray, shots: int = 1024):
        qc = self._build_circuit(rotation_params, entangle_params)
        job = execute(qc, backend, shots=shots)
        return job.result().get_counts(qc)


class QuantumFCL:
    """Parameterised quantum circuit emulating a fully‑connected layer."""
    def __init__(self, n_qubits: int, backend, shots: int):
        self._circuit = QuantumCircuit(n_qubits)
        self.theta = ParameterVector("theta", 1)
        self._circuit.h(range(n_qubits))
        self._circuit.barrier()
        self._circuit.ry(self.theta[0], range(n_qubits))
        self._circuit.measure_all()
        self.backend = backend
        self.shots = shots

    def run(self, thetas):
        job = execute(
            self._circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[{self.theta[0]: theta} for theta in thetas],
        )
        result = job.result().get_counts(self._circuit)
        counts = np.array(list(result.values()))
        states = np.array(list(result.keys())).astype(float)
        probabilities = counts / self.shots
        expectation = np.sum(states * probabilities)
        return np.array([expectation])


class HybridClassifierQuantum:
    """Hybrid quantum classifier with data encoding, variational ansatz, optional self‑attention and FCL."""
    def __init__(self,
                 num_qubits: int,
                 depth: int,
                 use_attention: bool = True,
                 attention_dim: int = 4,
                 use_fcl: bool = True,
                 backend=None,
                 shots: int = 1024):
        self.num_qubits = num_qubits
        self.depth = depth
        self.use_attention = use_attention
        self.use_fcl = use_fcl
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots

        # Data encoding parameters
        self.encoding = ParameterVector("x", num_qubits)

        # Variational parameters
        self.weights = ParameterVector("theta", num_qubits * depth)

        # Optional self‑attention
        if use_attention:
            self.attention = QuantumSelfAttention(n_qubits=attention_dim)
        else:
            self.attention = None

        # Optional fully‑connected emulation
        if use_fcl:
            self.fcl = QuantumFCL(n_qubits=1, backend=self.backend, shots=self.shots)
        else:
            self.fcl = None

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)
        # Data encoding
        for param, qubit in zip(self.encoding, range(self.num_qubits)):
            qc.rx(param, qubit)

        # Variational ansatz
        idx = 0
        for _ in range(self.depth):
            for qubit in range(self.num_qubits):
                qc.ry(self.weights[idx], qubit)
                idx += 1
            for qubit in range(self.num_qubits - 1):
                qc.cz(qubit, qubit + 1)

        # Optional self‑attention entanglement
        if self.use_attention:
            # Random parameters for demonstration; in practice these would be trainable
            rotation_params = np.random.rand(3 * self.attention.n_qubits)
            entangle_params = np.random.rand(self.attention.n_qubits - 1)
            attention_circuit = self.attention._build_circuit(rotation_params, entangle_params)
            qc.compose(attention_circuit, inplace=True)

        # Measurement
        for i in range(self.num_qubits):
            qc.measure(i, i)
        return qc

    def run(self, x: np.ndarray) -> np.ndarray:
        """Execute the hybrid circuit and return classification logits."""
        # Bind data encoding parameters to input features
        param_binds = [{self.encoding[i]: val} for i, val in enumerate(x)]
        qc = self._build_circuit()
        job = execute(qc, self.backend, shots=self.shots, parameter_binds=param_binds)
        result = job.result()
        counts = result.get_counts(qc)

        # Convert counts to expectation value (simple parity)
        expectation = 0.0
        for state, count in counts.items():
            parity = (-1) ** state.count('1')
            expectation += parity * count
        expectation /= self.shots

        # Optional FCL measurement
        if self.use_fcl:
            fcl_expect = self.fcl.run([expectation])
            expectation += fcl_expect[0]
        return np.array([expectation])

    @staticmethod
    def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, list[ParameterVector], list[ParameterVector], list[SparsePauliOp]]:
        """Return circuit, encoding params, variational params, observables."""
        qc_obj = HybridClassifierQuantum(num_qubits, depth)
        qc = qc_obj._build_circuit()
        encoding = [qc_obj.encoding]
        weights = [qc_obj.weights]
        observables = [SparsePauliOp("Z")]
        return qc, encoding, weights, observables


__all__ = ["HybridClassifierQuantum"]
