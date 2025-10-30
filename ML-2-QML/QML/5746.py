"""HybridFastEstimator with quantum hybrid head using Qiskit.

This module extends the classical HybridFastEstimator by replacing the
output head with a quantum circuit that evaluates expectation values
of a parameterised variational circuit.  The estimator remains fully
batched and supports shot‑noise simulation.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Iterable, List, Sequence, Callable, Union

from qiskit import QuantumCircuit, transpile, assemble
from qiskit.providers.aer import AerSimulator
from qiskit.quantum_info import Statevector, Pauli

# Import the classical base estimator
from hybrid_fast_estimator import HybridFastEstimator as _BaseEstimator
from hybrid_fast_estimator import HybridHead as _HybridHead
from hybrid_fast_estimator import ClassicalHybridHead as _ClassicalHybridHead

class QuantumHybridHead(_HybridHead):
    """Hybrid head that evaluates a parametrised quantum circuit to produce a probability."""
    def __init__(
        self,
        n_qubits: int,
        backend_name: str = "aer_simulator",
        shots: int = 1024,
        shift: float = np.pi / 2
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.backend = AerSimulator()
        self.shots = shots
        self.shift = shift
        self._circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        qc.h(range(self.n_qubits))
        qc.barrier()
        theta = qc.parameters[0]
        qc.ry(theta, range(self.n_qubits))
        qc.measure_all()
        return qc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is expected to be a 1‑D tensor of angles
        angles = x.detach().cpu().numpy().flatten()
        probs = []
        pauli_z = Pauli('Z' + 'I' * (self.n_qubits - 1))
        for angle in angles:
            bound_qc = self._circuit.assign_parameters(
                {self._circuit.parameters[0]: angle}, inplace=False
            )
            state = Statevector.from_instruction(bound_qc)
            expectation = float(state.expectation_value(pauli_z))
            probs.append(expectation)
        return torch.tensor(probs, dtype=torch.float32, device=x.device)

class HybridFastEstimator(_BaseEstimator):
    """Hybrid estimator that uses a quantum head."""
    def __init__(self, model: nn.Module, head: QuantumHybridHead | None = None):
        super().__init__(model, head=head or QuantumHybridHead(model.out_features))
