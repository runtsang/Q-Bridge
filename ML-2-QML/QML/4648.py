"""Hybrid quantum classifier mirroring the classical interface.

The module builds a variational circuit with data‑encoding and
entangling layers, and forwards samples through a parameterised
Ansatz.  The resulting expectation values are returned as logits
for two classes.  The implementation uses Qiskit + Aer and is
fully compatible with PyTorch data pipelines.
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Iterable, Tuple, List
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

# --------------------------------------------------------------------------- #
# Quantum circuit builder – data‑encoding + entangling layers
# --------------------------------------------------------------------------- #

def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, List[ParameterVector], List[ParameterVector], List[SparsePauliOp]]:
    """
    Construct a layered ansatz that mirrors the classical network:
    - an RX encoding of the input data,
    - a stack of RX rotations and C‑Z entangling gates.
    Returns the circuit and metadata required for the hybrid interface.
    """
    encoding = ParameterVector("x", length=num_qubits)
    weights = ParameterVector("theta", length=num_qubits * depth)

    qc = QuantumCircuit(num_qubits)

    # Data‑encoding
    for idx, qubit in enumerate(range(num_qubits)):
        qc.rx(encoding[idx], qubit)

    # Variational layers
    w_idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            qc.ry(weights[w_idx], qubit)
            w_idx += 1
        for qubit in range(num_qubits - 1):
            qc.cz(qubit, qubit + 1)

    # Observables – one Z per qubit
    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]

    return qc, list(encoding), list(weights), observables

# --------------------------------------------------------------------------- #
# Hybrid quantum classifier – forward pass using Aer simulator
# --------------------------------------------------------------------------- #

class HybridQuantumClassifier:
    """
    Quantum classifier exposing the same interface as its classical
    counterpart.  The forward method accepts a batch of input angles
    and returns log‑likelihoods for two classes derived from the
    expectation values of Z observables.
    """
    def __init__(self, num_qubits: int, depth: int = 2, shots: int = 1024):
        self.num_qubits = num_qubits
        self.shots = shots
        self.circuit, self.encoding, self.weights, self.observables = build_classifier_circuit(
            num_qubits, depth
        )
        self.backend = Aer.get_backend("qasm_simulator")

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Execute the variational circuit for each sample in the batch,
        bind the encoding parameters, and compute the expectation of
        the Z observables.  The two logits are obtained by summing
        the two largest expectation values (positive/negative)
        to emulate a binary classification output.
        """
        batch_size = states.shape[0]
        logits = []
        for i in range(batch_size):
            param_binds = {self.encoding[j]: states[i, j].item() for j in range(self.num_qubits)}
            job = execute(self.circuit, self.backend, shots=self.shots, parameter_binds=[param_binds])
            result = job.result()
            counts = result.get_counts(self.circuit)
            probs = {state: count / self.shots for state, count in counts.items()}

            # Compute expectation of each Z observable
            exp_vals = []
            for idx in range(self.num_qubits):
                exp = sum((1 - 2 * int(state[::-1][idx])) * p for state, p in probs.items())
                exp_vals.append(exp)

            # Two‑class logits: use mean of positives vs negatives
            pos = sum(max(v, 0) for v in exp_vals)
            neg = -sum(min(v, 0) for v in exp_vals)
            logits.append([pos, neg])

        return torch.tensor(logits, dtype=torch.float32)

__all__ = ["HybridQuantumClassifier"]
