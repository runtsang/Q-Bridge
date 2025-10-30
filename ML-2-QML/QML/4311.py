"""Quantum hybrid sampler that mirrors the classical HybridSamplerQNN.

The implementation uses Qiskit to build a parameterized circuit for sampling,
a quantum kernel based on state‑vector overlap, and a simple fully‑connected
layer realized as a single‑qubit rotation.  The class exposes the same
public API as the classical version so that downstream code can swap
between the two without modification.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.providers.aer import AerSimulator
from qiskit.quantum_info import Statevector
from qiskit.utils import QuantumInstance
from typing import Iterable, Sequence


class QuantumSamplerQNN:
    """
    Quantum sampler that implements a small variational circuit.

    Parameters
    ----------
    input_dim : int
        Number of input parameters (qubits).
    hidden_dim : int
        Number of variational parameters per qubit.
    """

    def __init__(self, input_dim: int = 2, hidden_dim: int = 4) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.qc = QuantumCircuit(input_dim)
        self._build_circuit()
        self.backend = AerSimulator()
        self.qi = QuantumInstance(self.backend, shots=1024)

    def _build_circuit(self) -> None:
        """Build a parameterized circuit that mirrors the classical sampler."""
        self.input_params = ParameterVector("x", self.input_dim)
        self.weight_params = ParameterVector("w", self.hidden_dim)

        # Encode inputs
        for i in range(self.input_dim):
            self.qc.ry(self.input_params[i], i)

        # Entangling layer
        self.qc.cx(0, 1)

        # Variational layer
        for i in range(self.hidden_dim):
            self.qc.ry(self.weight_params[i], i % self.input_dim)

        # Final entangling
        self.qc.cx(0, 1)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Return a probability distribution using a state‑vector simulation."""
        bound_circuit = self.qc.bind_parameters(
            {self.input_params[i]: val for i, val in enumerate(inputs)}
        )
        result = self.qi.execute(bound_circuit)
        counts = result.get_counts()
        probs = np.zeros(self.input_dim)
        total = sum(counts.values())
        for state, c in counts.items():
            probs[int(state, 2)] = c / total
        return probs

    def sample(self, inputs: np.ndarray, num_samples: int = 1) -> np.ndarray:
        """Draw samples from the quantum circuit."""
        probs = self.forward(inputs)
        return np.random.choice(self.input_dim, size=num_samples, p=probs)

    def kernel_matrix(
        self,
        a: Sequence[np.ndarray],
        b: Sequence[np.ndarray],
    ) -> np.ndarray:
        """Compute a quantum kernel via state‑vector overlap."""
        def state_vector(inp: np.ndarray) -> np.ndarray:
            bound = self.qc.bind_parameters(
                {self.input_params[i]: val for i, val in enumerate(inp)}
            )
            sv = Statevector(bound)
            return sv.data

        gram = np.zeros((len(a), len(b)))
        for i, x in enumerate(a):
            sv_x = state_vector(x)
            for j, y in enumerate(b):
                sv_y = state_vector(y)
                gram[i, j] = np.abs(np.vdot(sv_x, sv_y)) ** 2
        return gram

    def run_fcl(self, thetas: Iterable[float]) -> np.ndarray:
        """Simulate a single‑qubit fully‑connected layer via a rotation."""
        theta = float(next(iter(thetas)))
        qc = QuantumCircuit(1)
        qc.ry(theta, 0)
        result = self.qi.execute(qc)
        counts = result.get_counts()
        total = sum(counts.values())
        probs = np.array([counts.get("0", 0), counts.get("1", 0)]) / total
        expectation = probs[1] - probs[0]
        return np.array([expectation])


def SamplerQNN() -> QuantumSamplerQNN:
    """Return a ready‑to‑use instance of the quantum hybrid sampler."""
    return QuantumSamplerQNN()


__all__ = ["QuantumSamplerQNN", "SamplerQNN"]
