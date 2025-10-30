"""Quantum‑hybrid classifier – quantum branch.

This module implements the quantum side of the hybrid classifier.
It mirrors the classical implementation but replaces the neural net
with a variational circuit executed on a Qiskit Aer simulator.
"""

import numpy as np
import torch
import torch.nn as nn
from qiskit import QuantumCircuit, Aer, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Pauli
from typing import Iterable, Tuple, List

def generate_superposition_data(num_qubits: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic quantum states of the form
    cos(theta)|0...0> + exp(i phi) sin(theta)|1...1>.
    The labels are a non‑linear function of theta and phi.
    """
    omega_0 = np.zeros(2 ** num_qubits, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_qubits, dtype=complex)
    omega_1[-1] = 1.0

    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2 ** num_qubits), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
    labels = np.sign(np.sin(2 * thetas) * np.cos(phis))
    return states, labels.astype(np.float32)

class QuantumHybridClassifier:
    """
    Quantum classifier that encodes classical data into a quantum
    state, applies a depth‑controlled variational ansatz and
    measures Z on each qubit.  The measurement outcomes are
    passed through a classical linear head to produce logits.
    """

    def __init__(self, num_qubits: int, depth: int = 2):
        self.num_qubits = num_qubits
        self.depth = depth
        self.circuit, self.encoding, self.weights, self.observables = self.build_classifier_circuit(num_qubits, depth)
        self.head = nn.Linear(num_qubits, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.  `x` is a tensor of shape (batch, num_qubits)
        containing rotation angles for the data‑encoding layer.
        Returns logits of shape (batch, 2).
        """
        batch_size = x.shape[0]
        backend = Aer.get_backend('qasm_simulator')
        shots = 1024
        all_logits = []
        for i in range(batch_size):
            qc = self.circuit.copy()
            param_bind = {p: float(x[i, j]) for j, p in enumerate(self.encoding)}
            qc = qc.bind_parameters(param_bind)
            job = backend.run(transpile(qc, backend), shots=shots)
            result = job.result()
            counts = result.get_counts(qc)
            exp_vals = []
            for qubit in range(self.num_qubits):
                exp = 0.0
                for bitstring, count in counts.items():
                    bit = bitstring[self.num_qubits - 1 - qubit]
                    exp += (1.0 if bit == '0' else -1.0) * count
                exp /= shots
                exp_vals.append(exp)
            exp_tensor = torch.tensor(exp_vals, dtype=torch.float32, device=x.device)
            logits = self.head(exp_tensor)
            all_logits.append(logits)
        return torch.stack(all_logits)

    @staticmethod
    def build_classifier_circuit(num_qubits: int,
                                 depth: int = 2) -> Tuple[QuantumCircuit, List[ParameterVector], List[ParameterVector], List[Pauli]]:
        """
        Construct a layered ansatz with explicit data‑encoding and
        variational parameters.
        """
        encoding = ParameterVector("x", num_qubits)
        weights = [ParameterVector(f"theta_{d}", num_qubits) for d in range(depth)]
        qc = QuantumCircuit(num_qubits)
        for qubit, param in enumerate(encoding):
            qc.rx(param, qubit)
        for w in weights:
            for qubit, param in enumerate(w):
                qc.ry(param, qubit)
            for qubit in range(num_qubits - 1):
                qc.cz(qubit, qubit + 1)
        observables = [Pauli('Z') for _ in range(num_qubits)]
        return qc, encoding, weights, observables

__all__ = ["QuantumHybridClassifier", "build_classifier_circuit", "generate_superposition_data"]
