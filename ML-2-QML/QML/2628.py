"""Quantum primitives for hybrid training.

Provides the core quantum circuit used by the PyTorch model and a
parameterised sampler that can be used as an alternative head.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import assemble, transpile
from qiskit.circuit import ParameterVector
from qiskit.primitives import Sampler as QiskitSampler


class QuantumCircuit:
    """Parametrised two‑qubit circuit executed on Aer."""

    def __init__(self, n_qubits: int, backend, shots: int) -> None:
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        all_qubits = list(range(n_qubits))
        self.theta = qiskit.circuit.Parameter("theta")

        self._circuit.h(all_qubits)
        self._circuit.barrier()
        self._circuit.ry(self.theta, all_qubits)
        self._circuit.measure_all()

        self.backend = backend
        self.shots = shots

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self._circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()

        def expectation(count_dict):
            counts = np.array(list(count_dict.values()))
            states = np.array(list(count_dict.keys())).astype(float)
            probabilities = counts / self.shots
            return np.sum(states * probabilities)

        if isinstance(result, list):
            return np.array([expectation(item) for item in result])
        return np.array([expectation(result)])


def build_sampler_qnn(n_qubits: int, backend, shots: int) -> qiskit_machine_learning.neural_networks.SamplerQNN:
    """Return a qiskit‑machine‑learning SamplerQNN object.

    The sampler uses a simple two‑qubit circuit with
    input and weight parameters.  It can be used as a
    differentiable head in a hybrid model.
    """
    from qiskit.circuit import ParameterVector
    from qiskit_machine_learning.neural_networks import SamplerQNN
    from qiskit.primitives import StatevectorSampler as Sampler

    inputs = ParameterVector("input", 2)
    weights = ParameterVector("weight", 4)

    qc = qiskit.QuantumCircuit(n_qubits)
    qc.ry(inputs[0], 0)
    qc.ry(inputs[1], 1)
    qc.cx(0, 1)
    qc.ry(weights[0], 0)
    qc.ry(weights[1], 1)
    qc.cx(0, 1)
    qc.ry(weights[2], 0)
    qc.ry(weights[3], 1)

    sampler = Sampler()
    return SamplerQNN(circuit=qc, input_params=inputs, weight_params=weights, sampler=sampler)


__all__ = ["QuantumCircuit", "build_sampler_qnn"]
