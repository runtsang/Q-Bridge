"""Quantum sampler network that mirrors the classical transformer.

The quantum circuit accepts a 2‑dimensional input vector and a 4‑parameter
weight vector.  It consists of a Ry‑CX entangling block followed by a
random two‑qubit circuit and a second Ry‑CX block.  The StatevectorSampler
provides the probability distribution over the two basis states, which is
returned as a 2‑dimensional probability vector.  The class exposes a
`forward` method compatible with PyTorch tensors, making it a drop‑in
replacement for the classical SamplerQNNGen150 in hybrid training loops.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import qiskit
from qiskit import QuantumCircuit, transpile, assemble
from qiskit.circuit import ParameterVector
from qiskit.providers.aer import AerSimulator
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import StatevectorSampler

class SamplerQNNGen150:
    """Quantum sampler that outputs a probability vector over two classes."""
    def __init__(self, n_qubits: int = 2, depth: int = 2):
        # Build a parameterised circuit
        self.inputs = ParameterVector("input", n_qubits)
        self.weights = ParameterVector("weight", n_qubits * depth)
        self.circuit = QuantumCircuit(n_qubits)
        # Input layer
        for i in range(n_qubits):
            self.circuit.ry(self.inputs[i], i)
        # First entangling block
        self.circuit.cx(0, 1)
        # Random entanglement
        self.circuit += qiskit.circuit.random.random_circuit(n_qubits, depth)
        # Weight layers
        for d in range(depth):
            for i in range(n_qubits):
                self.circuit.ry(self.weights[d * n_qubits + i], i)
            if d < depth - 1:
                self.circuit.cx(0, 1)
        self.circuit.measure_all()
        # Sampler primitive
        self.backend = AerSimulator()
        self.sampler = StatevectorSampler(self.backend)
        self.sampler_qnn = SamplerQNN(circuit=self.circuit,
                                      input_params=self.inputs,
                                      weight_params=self.weights,
                                      sampler=self.sampler)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return a 2‑dimensional probability vector."""
        # The qiskit SamplerQNN accepts a torch.Tensor of shape (batch, 2)
        probs = self.sampler_qnn(inputs)
        # probs is a torch.Tensor of shape (batch, 2**n_qubits)
        # Collapse to two probabilities by summing over the first qubit
        if probs.ndim == 2 and probs.size(1) == 4:
            p0 = probs[:, 0] + probs[:, 2]
            p1 = probs[:, 1] + probs[:, 3]
            return torch.stack([p0, p1], dim=1)
        return probs

__all__ = ["SamplerQNNGen150"]
