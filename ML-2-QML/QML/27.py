"""Quantum sampler network based on a Pennylane variational circuit.

The class builds a parameterized circuit with a configurable number of qubits and layers.
It exposes a ``forward`` method that returns the probability distribution over the computational basis
for a given classical input vector.  The implementation uses Qiskit’s StatevectorSampler for
exact probability extraction and Pennylane for auto‑differentiation.
"""

from __future__ import annotations

from typing import Iterable

import pennylane as qml
from pennylane import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import Statevector
from qiskit.circuit import ParameterVector


class SamplerQNN:
    """
    Quantum sampler with a variational circuit.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    depth : int
        Number of variational layers.
    seed : int | None
        Random seed for parameter initialization.
    """

    def __init__(self, num_qubits: int = 2, depth: int = 2, seed: int | None = None) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.seed = seed
        self.dev = qml.device("qiskit.aer.statevector_simulator", wires=num_qubits)
        self._build_circuit()
        self.sampler = StatevectorSampler()

    def _build_circuit(self) -> None:
        """Constructs a variational circuit with Ry rotations and entangling CNOTs."""
        self.params = ParameterVector("theta", self.depth * self.num_qubits)
        self.input_params = ParameterVector("x", self.num_qubits)

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(inputs: Iterable[float], params: Iterable[float]) -> np.ndarray:
            # Apply input rotations
            for i in range(self.num_qubits):
                qml.RY(inputs[i], wires=i)
            # Variational layers
            idx = 0
            for _ in range(self.depth):
                for i in range(self.num_qubits):
                    qml.RY(params[idx], wires=i)
                    idx += 1
                # Entangle
                for i in range(self.num_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            return qml.probs(wires=range(self.num_qubits))

        self.circuit = circuit

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Computes the probability distribution for each input in the batch.

        Args:
            inputs: Tensor of shape (batch, num_qubits) with values in [0, π].

        Returns:
            Tensor of shape (batch, 2**num_qubits) containing probabilities.
        """
        probs = []
        for inp in inputs:
            probs.append(self.circuit(inp.detach().numpy(), self.params))
        return torch.tensor(probs, dtype=torch.float)

    def sample(self, inputs: torch.Tensor, num_shots: int = 1024) -> torch.Tensor:
        """
        Draws samples from the quantum circuit using the StatevectorSampler backend.

        Args:
            inputs: Tensor of shape (batch, num_qubits).
            num_shots: Number of measurement shots per input.

        Returns:
            Tensor of shape (batch, 2**num_qubits) with sample counts.
        """
        counts = []
        for inp in inputs:
            state = self.circuit(inp.detach().numpy(), self.params)
            sv = Statevector(state)
            sample_counts = self.sampler.run(sv, shots=num_shots).get_counts()
            # Convert counts dict to probability vector
            vec = np.zeros(2 ** self.num_qubits)
            for bitstring, cnt in sample_counts.items():
                idx = int(bitstring, 2)
                vec[idx] = cnt / num_shots
            counts.append(vec)
        return torch.tensor(counts, dtype=torch.float)

__all__ = ["SamplerQNN"]
