"""Hybrid self‑attention module with quantum‑centric implementations.

The class mirrors the classical version but replaces each sub‑module with a
Qiskit‑based variational circuit or neural network.  It keeps the original
anchor API (`SelfAttention()`) while offering genuine quantum experiments for
attention, sampling, a fully‑connected layer and an estimator.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.circuit import ParameterVector, Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import SamplerQNN as QSamplerQNN
from qiskit_machine_learning.neural_networks import EstimatorQNN as QEstimatorQNN
from qiskit.primitives import StatevectorSampler, StatevectorEstimator
from typing import Iterable

class HybridSelfAttention:
    """
    Quantum hybrid self‑attention that integrates four reference modules:
    - Quantum attention circuit
    - Quantum sampler QNN
    - Quantum fully‑connected layer
    - Quantum estimator
    """

    def __init__(self, embed_dim: int = 4, n_features: int = 1,
                 backend=None, shots: int = 1024) -> None:
        self.embed_dim = embed_dim
        self.n_features = n_features
        self.shots = shots
        self.backend = backend or Aer.get_backend("qasm_simulator")

        # Random parameters for the attention circuit
        self.rotation_params = np.random.randn(embed_dim * 3).astype(np.float32)
        self.entangle_params = np.random.randn(embed_dim - 1).astype(np.float32)

        # Quantum modules
        self.attention = self._build_quantum_attention()
        self.sampler = self._build_sampler_qnn()
        self.fcl = self._build_quantum_fcl()
        self.estimator = self._build_estimator_qnn()

    # ---------- Quantum components ----------
    def _build_quantum_attention(self):
        class QuantumSelfAttention:
            def __init__(self, n_qubits: int, backend, shots: int,
                         rotation_params: np.ndarray, entangle_params: np.ndarray):
                self.n_qubits = n_qubits
                self.backend = backend
                self.shots = shots
                self.rotation_params = rotation_params
                self.entangle_params = entangle_params

            def _build_circuit(self) -> QuantumCircuit:
                qr = QuantumRegister(self.n_qubits, "q")
                cr = ClassicalRegister(self.n_qubits, "c")
                circuit = QuantumCircuit(qr, cr)
                for i in range(self.n_qubits):
                    circuit.rx(self.rotation_params[3 * i], i)
                    circuit.ry(self.rotation_params[3 * i + 1], i)
                    circuit.rz(self.rotation_params[3 * i + 2], i)
                for i in range(self.n_qubits - 1):
                    circuit.crx(self.entangle_params[i], i, i + 1)
                circuit.measure(qr, cr)
                return circuit

            def run(self) -> dict:
                circuit = self._build_circuit()
                job = execute(circuit, self.backend, shots=self.shots)
                return job.result().get_counts(circuit)

        return QuantumSelfAttention(self.embed_dim, self.backend, self.shots,
                                    self.rotation_params, self.entangle_params)

    def _build_sampler_qnn(self):
        inputs2 = ParameterVector("input", 2)
        weights2 = ParameterVector("weight", 4)
        qc2 = QuantumCircuit(2)
        qc2.ry(inputs2[0], 0)
        qc2.ry(inputs2[1], 1)
        qc2.cx(0, 1)
        qc2.ry(weights2[0], 0)
        qc2.ry(weights2[1], 1)
        qc2.cx(0, 1)
        qc2.ry(weights2[2], 0)
        qc2.ry(weights2[3], 1)
        sampler = StatevectorSampler()
        return QSamplerQNN(circuit=qc2,
                           input_params=inputs2,
                           weight_params=weights2,
                           sampler=sampler)

    def _build_quantum_fcl(self):
        class QuantumFCL:
            def __init__(self, n_qubits: int, backend, shots: int):
                self._circuit = QuantumCircuit(n_qubits)
                self.theta = Parameter("theta")
                self._circuit.h(range(n_qubits))
                self._circuit.barrier()
                self._circuit.ry(self.theta, range(n_qubits))
                self._circuit.measure_all()
                self.backend = backend
                self.shots = shots

            def run(self, thetas: Iterable[float]) -> np.ndarray:
                job = execute(self._circuit,
                              self.backend,
                              shots=self.shots,
                              parameter_binds=[{self.theta: t} for t in thetas])
                result = job.result().get_counts(self._circuit)
                counts = np.array(list(result.values()))
                states = np.array([int(k, 2) for k in result.keys()], dtype=float)
                probs = counts / self.shots
                expectation = np.sum(states * probs)
                return np.array([expectation])

        return QuantumFCL(1, self.backend, self.shots)

    def _build_estimator_qnn(self):
        params1 = [Parameter("input1"), Parameter("weight1")]
        qc1 = QuantumCircuit(1)
        qc1.h(0)
        qc1.ry(params1[0], 0)
        qc1.rx(params1[1], 0)
        observable1 = SparsePauliOp.from_list([("Y", 1)])
        estimator = StatevectorEstimator()
        return QEstimatorQNN(circuit=qc1,
                             observables=observable1,
                             input_params=[params1[0]],
                             weight_params=[params1[1]],
                             estimator=estimator)

    # ---------- Public API ----------
    def run_attention(self) -> dict:
        """Execute the quantum attention circuit and return measurement counts."""
        return self.attention.run()

    def run_sampler(self) -> np.ndarray:
        """Sample from the quantum sampler QNN."""
        return self.sampler.run()

    def run_fcl(self, thetas: Iterable[float]) -> np.ndarray:
        """Compute the expectation of the quantum fully‑connected layer."""
        return self.fcl.run(thetas)

    def run_estimator(self, input_val: float, weight_val: float) -> np.ndarray:
        """Estimate an expectation value with the quantum estimator."""
        return self.estimator.run(input_params=[input_val], weight_params=[weight_val])

    def run(self) -> dict:
        """Return a dictionary of all four component outputs."""
        return {
            "attention": self.run_attention(),
            "sampler": self.run_sampler(),
            "fcl": self.run_fcl([0.0, 1.0, 2.0]),
            "estimator": self.run_estimator(0.5, 1.0),
        }

def SelfAttention() -> HybridSelfAttention:
    """Factory that returns a HybridSelfAttention instance with default settings."""
    return HybridSelfAttention(embed_dim=4)

__all__ = ["SelfAttention"]
