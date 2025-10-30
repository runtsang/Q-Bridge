"""Quantum sampler with multi‑layer entangling circuit and StatevectorSampler."""

from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN as QSamplerQNN
from qiskit.primitives import StatevectorSampler


class SamplerQNNClass:
    """A variational SamplerQNN that exposes a StatevectorSampler."""

    def __init__(
        self,
        input_dim: int = 2,
        hidden_layers: int = 3,
        qubits: int = 2,
    ) -> None:
        self.input_params = ParameterVector("input", input_dim)
        # Two Ry parameters per hidden layer on the first qubit
        self.weight_params = ParameterVector("weight", hidden_layers * 2)
        self.circuit = self._build_circuit(qubits)
        self.sampler = StatevectorSampler()
        self.qsampler = QSamplerQNN(
            circuit=self.circuit,
            input_params=self.input_params,
            weight_params=self.weight_params,
            sampler=self.sampler,
        )

    def _build_circuit(self, qubits: int) -> QuantumCircuit:
        qc = QuantumCircuit(qubits)

        # Input rotations
        for i in range(qubits):
            qc.ry(self.input_params[i], i)

        # First entangling layer
        for i in range(qubits - 1):
            qc.cx(i, i + 1)

        # Parameterised rotations on the first qubit
        for w in self.weight_params:
            qc.ry(w, 0)

        # Second entangling layer
        for i in range(qubits - 1):
            qc.cx(i, i + 1)

        return qc

    def sample(self, nshots: int = 1024) -> dict:
        """Return a measurement histogram using the underlying StatevectorSampler."""
        return self.qsampler.sample(nshots=nshots)

    def get_circuit(self) -> QuantumCircuit:
        """Return the raw quantum circuit for inspection or export."""
        return self.circuit

    def get_params(self) -> tuple[ParameterVector, ParameterVector]:
        """Return the input and weight parameter vectors."""
        return self.input_params, self.weight_params


def SamplerQNN() -> SamplerQNNClass:
    """Factory that keeps the original anchor signature."""
    return SamplerQNNClass()


__all__ = ["SamplerQNNClass", "SamplerQNN"]
