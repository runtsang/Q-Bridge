"""Hybrid quantum graph neural network with a sampler circuit.

This module mirrors the classical implementation above but uses
Qiskit to build a parameterised quantum circuit that serves as the
sampler.  The core feed‑forward logic is identical to the original
`GraphQNN` module: each layer is a unitary that acts on the current
state and discards the unused qubits.  The final layer is replaced by
a small variational sampler that produces measurement probabilities
over two outcomes.

The class exposes a `sample` method that accepts classical input
values, binds them to the circuit, and returns the resulting
probability distribution.  It also re‑exports the graph utilities
from the original QNN module so that fidelity‑based adjacency graphs
can be constructed directly from the output states.
"""

from __future__ import annotations

import itertools
from typing import Iterable, Sequence

import networkx as nx
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN as QSamplerQNN
from qiskit.primitives import StatevectorSampler as Sampler

from.GraphQNN import (
    random_network,
    feedforward,
    state_fidelity,
    fidelity_adjacency,
)


class HybridQuantumGraphQNN:
    """
    Quantum hybrid GNN with a variational sampler.

    Parameters
    ----------
    qnn_arch : Sequence[int]
        Layer sizes of the underlying unitary network.
    sampler_hidden : int, default 4
        Number of parameters in the sampler circuit.
    """

    def __init__(self, qnn_arch: Sequence[int], sampler_hidden: int = 4) -> None:
        self.qnn_arch = list(qnn_arch)

        # Build the unitary network
        self.arch, self.unitaries, self.training_data, self.target_unitary = random_network(
            self.qnn_arch, samples=10
        )

        # Build the sampler circuit
        self.inputs = ParameterVector("input", 2)
        self.weights = ParameterVector("weight", sampler_hidden * 2)

        self.qc = QuantumCircuit(2)
        for i in range(2):
            self.qc.ry(self.inputs[i], i)
        for i in range(sampler_hidden):
            self.qc.ry(self.weights[i * 2], 0)
            self.qc.ry(self.weights[i * 2 + 1], 1)
        self.qc.cx(0, 1)

        # Sampler primitive
        self.sampler = Sampler()
        self.sampler_qnn = QSamplerQNN(
            circuit=self.qc,
            input_params=self.inputs,
            weight_params=self.weights,
            sampler=self.sampler,
        )

    def sample(self, input_vals: Sequence[float]) -> dict:
        """
        Execute the sampler circuit for a given input vector.

        Parameters
        ----------
        input_vals : Sequence[float]
            Two real numbers that bind to the `input` parameters.

        Returns
        -------
        dict
            Mapping from measurement outcome to probability.
        """
        bound_qc = self.qc.bind_parameters(
            {self.inputs[i]: input_vals[i] for i in range(2)}
        )
        result = self.sampler.sample(bound_qc, shots=1024)
        counts = result.get_counts()
        total = sum(counts.values())
        return {k: v / total for k, v in counts.items()}

    @staticmethod
    def feedforward(
        qnn_arch: Sequence[int],
        unitaries: Sequence[Sequence[qiskit.quantum_info.QubitStateVector]],
        samples: Iterable[tuple[qiskit.quantum_info.Statevector, qiskit.quantum_info.Statevector]],
    ):
        return feedforward(qnn_arch, unitaries, samples)

    # Re‑export graph utilities
    state_fidelity = staticmethod(state_fidelity)
    fidelity_adjacency = staticmethod(fidelity_adjacency)


__all__ = ["HybridQuantumGraphQNN"]
