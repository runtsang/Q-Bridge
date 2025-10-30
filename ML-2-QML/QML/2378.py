"""Hybrid fully‑connected layer with a quantum sampler.

This quantum implementation combines the FCL rotation and a SamplerQNN
parameterized circuit into a single 3‑qubit circuit.  The first qubit
encodes the fully‑connected rotation, while the remaining two qubits
implement the sampler.  The `run` method accepts a list of seven
parameters and returns both the expectation value of qubit‑0 and a
probability distribution over the two sampler qubits."""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, execute
from qiskit.circuit import Parameter, ParameterVector


def HybridFCLSampler(backend, shots: int = 1024):
    class _HybridFCLSampler:
        def __init__(self) -> None:
            # Parameters
            self.theta = Parameter("theta")
            self.inputs = ParameterVector("input", 2)
            self.weights = ParameterVector("weight", 4)

            # Circuit
            self.circuit = QuantumCircuit(3)
            # Fully‑connected part on qubit 0
            self.circuit.h(0)
            self.circuit.ry(self.theta, 0)
            # Sampler part on qubits 1 and 2
            self.circuit.ry(self.inputs[0], 1)
            self.circuit.ry(self.inputs[1], 2)
            self.circuit.cx(1, 2)
            self.circuit.ry(self.weights[0], 1)
            self.circuit.ry(self.weights[1], 2)
            self.circuit.cx(1, 2)
            self.circuit.ry(self.weights[2], 1)
            self.circuit.ry(self.weights[3], 2)
            self.circuit.measure_all()

            self.backend = backend
            self.shots = shots

        def run(self, params: list[float]) -> Tuple[np.ndarray, np.ndarray]:
            """
            Parameters
            ----------
            params : list[float]
                Length‑7 list containing [theta, input0, input1, w0, w1, w2, w3].

            Returns
            -------
            Tuple[np.ndarray, np.ndarray]
                (expectation of qubit‑0, probability distribution of qubits‑1&2).
            """
            bind = {
                self.theta: params[0],
                self.inputs[0]: params[1],
                self.inputs[1]: params[2],
                self.weights[0]: params[3],
                self.weights[1]: params[4],
                self.weights[2]: params[5],
                self.weights[3]: params[6],
            }
            job = execute(
                self.circuit,
                self.backend,
                shots=self.shots,
                parameter_binds=[bind],
            )
            result = job.result()
            counts = result.get_counts(self.circuit)

            expectation = 0.0
            probs = np.zeros(4)  # 2 qubits -> 4 possible states
            for bitstring, cnt in counts.items():
                prob = cnt / self.shots
                # Expectation of qubit 0: +1 for '1', -1 for '0'
                expectation += (1 if bitstring[0] == "1" else -1) * prob
                # Probability of the two sampler qubits (reversed bit order)
                state = int(bitstring[1:][::-1], 2)
                probs[state] += prob

            return np.array([expectation]), probs

    return _HybridFCLSampler()


__all__ = ["HybridFCLSampler"]
