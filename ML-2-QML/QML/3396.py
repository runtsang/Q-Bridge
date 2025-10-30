from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit import ParameterVector


class QuantumSampler:
    """Parameterized quantum sampler that outputs a probability distribution."""
    def __init__(self) -> None:
        self.inputs = ParameterVector("input", 2)
        self.weights = ParameterVector("weight", 4)

        self.circuit = QuantumCircuit(2)
        self.circuit.ry(self.inputs[0], 0)
        self.circuit.ry(self.inputs[1], 1)
        self.circuit.cx(0, 1)
        self.circuit.ry(self.weights[0], 0)
        self.circuit.ry(self.weights[1], 1)
        self.circuit.cx(0, 1)
        self.circuit.ry(self.weights[2], 0)
        self.circuit.ry(self.weights[3], 1)

        self.backend = Aer.get_backend("statevector_simulator")

    def sample(self, input_vals: list[float], weight_vals: list[float], shots: int = 1024) -> np.ndarray:
        bound = {p: v for p, v in zip(self.inputs + self.weights, input_vals + weight_vals)}
        qc = self.circuit.bind_parameters(bound)
        job = execute(qc, self.backend, shots=shots)
        result = job.result().get_counts(qc)
        probs = np.array([result.get(k, 0) / shots for k in sorted(result)])
        return probs


class QuantumFullyConnected:
    """Quantum implementation of a fully‑connected layer using a single‑qubit Ry rotation."""
    def __init__(self, n_qubits: int = 1, shots: int = 1024) -> None:
        self.backend = Aer.get_backend("qasm_simulator")
        self.shots = shots

        self.circuit = QuantumCircuit(n_qubits)
        self.theta = ParameterVector("theta", n_qubits)
        self.circuit.h(range(n_qubits))
        self.circuit.ry(self.theta, range(n_qubits))
        self.circuit.measure_all()

    def run(self, thetas: list[float]) -> np.ndarray:
        bound = {p: t for p, t in zip(self.theta, thetas)}
        qc = self.circuit.bind_parameters(bound)
        job = execute(qc, self.backend, shots=self.shots)
        result = job.result().get_counts(qc)
        counts = np.array(list(result.values()))
        probs = counts / self.shots
        states = np.array([int(k, 2) for k in result.keys()])
        expectation = np.sum(states * probs)
        return np.array([expectation])


class HybridSamplerQNN:
    """Quantum‑side counterpart that mirrors the hybrid architecture in the classical module."""
    def __init__(self) -> None:
        self.sampler = QuantumSampler()
        self.fcl = QuantumFullyConnected()

    def sample(self, input_vals: list[float], weight_vals: list[float], thetas: list[float]) -> tuple[np.ndarray, np.ndarray]:
        probs = self.sampler.sample(input_vals, weight_vals)
        expectation = self.fcl.run(thetas)
        return probs, expectation


__all__ = ["HybridSamplerQNN"]
