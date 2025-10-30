"""Quantum circuits used by the hybrid classifier.

The module provides a two‑qubit expectation circuit for the hybrid
head and a 2‑qubit parameterised sampler circuit used by the SamplerQNN
module.  Both circuits are executed on the Aer simulator and expose a
`run` method that returns numpy arrays of expectation values or
probability vectors.
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, transpile, assemble
from qiskit.providers.aer import AerSimulator


class QuantumExpectationCircuit:
    """A two‑qubit expectation circuit used as a differentiable head."""
    def __init__(self, n_qubits: int = 2, shots: int = 200):
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = AerSimulator()
        self._circuit = QuantumCircuit(n_qubits)
        theta = qiskit.circuit.Parameter("theta")
        self.theta = theta
        self._circuit.h(range(n_qubits))
        self._circuit.ry(theta, range(n_qubits))
        self._circuit.measure_all()

    def run(self, thetas: list) -> np.ndarray:
        """Compute expectation of Z⊗I for each theta."""
        compiled = transpile(self._circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: t} for t in thetas],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        def expectation(count_dict):
            counts = np.array(list(count_dict.values()))
            states = np.array(list(count_dict.keys())).astype(float)
            probs = counts / self.shots
            return np.sum(states * probs)
        if isinstance(result, list):
            return np.array([expectation(r) for r in result])
        return np.array([expectation(result)])


class SamplerQNNCircuit:
    """2‑qubit parameterised sampler circuit returning a probability vector."""
    def __init__(self, shots: int = 200):
        self.shots = shots
        self.backend = AerSimulator()
        self._circuit = QuantumCircuit(2)
        # parameterised input angles
        self.input_params = qiskit.circuit.ParameterVector("input", 2)
        # parameterised weights
        self.weight_params = qiskit.circuit.ParameterVector("weight", 4)
        self._circuit.ry(self.input_params[0], 0)
        self._circuit.ry(self.input_params[1], 1)
        self._circuit.cx(0, 1)
        self._circuit.ry(self.weight_params[0], 0)
        self._circuit.ry(self.weight_params[1], 1)
        self._circuit.cx(0, 1)
        self._circuit.ry(self.weight_params[2], 0)
        self._circuit.ry(self.weight_params[3], 1)

    def run(self, inputs: list, weights: np.ndarray) -> np.ndarray:
        """Return a probability vector of shape (2,) for each input."""
        # Bind parameters
        param_binds = {
            self.input_params[0]: inputs[0],
            self.input_params[1]: inputs[1],
            self.weight_params[0]: weights[0],
            self.weight_params[1]: weights[1],
            self.weight_params[2]: weights[2],
            self.weight_params[3]: weights[3],
        }
        compiled = transpile(self._circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[param_binds],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        # Convert counts to probabilities for |00>, |01>, |10>, |11>
        probs = np.zeros(4)
        for state, cnt in result.items():
            idx = int(state, 2)
            probs[idx] = cnt / self.shots
        # Return marginal probability of measuring 0 on first qubit
        return np.array([probs[0] + probs[1], probs[2] + probs[3]])


__all__ = ["QuantumExpectationCircuit", "SamplerQNNCircuit"]
