"""Quantum implementation of HybridSamplerNet using Qiskit.
It mirrors the classical network but replaces each block with a
parameterised quantum circuit executed on an Aer simulator."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile, assemble, Aer, execute
from qiskit.providers.aer import AerSimulator
from qiskit.circuit import Parameter


class QuantumSelfAttention:
    """Basic quantum circuit representing a self‑attention style block."""

    def __init__(self, n_qubits: int = 4):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _build_circuit(
        self, rotation_params: np.ndarray, entangle_params: np.ndarray
    ) -> QuantumCircuit:
        circuit = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        circuit.measure_all()
        return circuit

    def run(
        self,
        backend: AerSimulator,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ) -> dict:
        circuit = self._build_circuit(rotation_params, entangle_params)
        transpiled = transpile(circuit, backend)
        qobj = assemble(transpiled, shots=shots)
        job = backend.run(qobj)
        return job.result().get_counts(circuit)


class SamplerQNN:
    """Parameterized two‑qubit sampler circuit."""

    def __init__(self):
        self.qr = QuantumRegister(2, "q")
        self.cr = ClassicalRegister(2, "c")
        self.circuit = QuantumCircuit(self.qr, self.cr)
        # Define parameters
        self.w0 = Parameter("w0")
        self.w1 = Parameter("w1")
        self.w2 = Parameter("w2")
        self.w3 = Parameter("w3")
        # Build circuit
        self.circuit.ry(self.w0, 0)
        self.circuit.ry(self.w1, 1)
        self.circuit.cx(0, 1)
        self.circuit.ry(self.w2, 0)
        self.circuit.ry(self.w3, 1)
        self.circuit.cx(0, 1)
        self.circuit.measure(self.qr, self.cr)

    def run(self, backend: AerSimulator, weights: np.ndarray, shots: int = 1024) -> np.ndarray:
        """Execute the sampler and return probabilities [p0, p1]."""
        bind_dict = {
            self.w0: weights[0],
            self.w1: weights[1],
            self.w2: weights[2],
            self.w3: weights[3],
        }
        transpiled = transpile(self.circuit, backend)
        qobj = assemble(transpiled, shots=shots, parameter_binds=[bind_dict])
        job = backend.run(qobj)
        result = job.result().get_counts(self.circuit)
        counts = np.array([result.get("00", 0), result.get("01", 0)])
        probs = counts / shots
        return probs


class ExpectationHead:
    """Single‑qubit circuit that returns the expectation value of Z after a ry rotation."""

    def __init__(self):
        self.qr = QuantumRegister(1, "q")
        self.cr = ClassicalRegister(1, "c")
        self.circuit = QuantumCircuit(self.qr, self.cr)
        self.theta = Parameter("theta")
        self.circuit.ry(self.theta, 0)
        self.circuit.measure(self.qr, self.cr)

    def run(self, backend: AerSimulator, theta: float, shots: int = 1024) -> float:
        bind_dict = {self.theta: theta}
        transpiled = transpile(self.circuit, backend)
        qobj = assemble(transpiled, shots=shots, parameter_binds=[bind_dict])
        job = backend.run(qobj)
        result = job.result().get_counts(self.circuit)
        p0 = result.get("0", 0) / shots
        p1 = result.get("1", 0) / shots
        return p0 - p1


class HybridSamplerNet:
    """Quantum implementation of the hybrid sampler network."""

    def __init__(self):
        self.backend = Aer.get_backend("aer_simulator")
        self.attention = QuantumSelfAttention(n_qubits=4)
        self.sampler = SamplerQNN()
        self.expectation = ExpectationHead()

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        batch_probs = []
        for sample in inputs:
            # Derive parameters for the attention block (illustrative only)
            rot_params = np.linspace(0, np.pi, 12)
            ent_params = np.linspace(0, np.pi / 2, 3)
            # Run attention (measures are not used further but demonstrate quantum interface)
            _ = self.attention.run(self.backend, rot_params, ent_params, shots=1024)
            # Map sample values to sampler weights
            weights = np.array([sample[0], sample[1], sample[0], sample[1]])
            probs = self.sampler.run(self.backend, weights, shots=1024)
            # Use the probability of outcome |01> as a parameter for the expectation head
            theta = probs[1] * np.pi
            expectation = self.expectation.run(self.backend, theta, shots=1024)
            # Convert expectation to a probability in [0, 1]
            prob_pos = (1 + expectation) / 2
            batch_probs.append([prob_pos, 1 - prob_pos])
        return np.array(batch_probs)


__all__ = ["HybridSamplerNet"]
