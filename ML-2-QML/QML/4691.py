"""Hybrid quantum classifier with a variational circuit and a classical post‑processing head.

The quantum part is a depth‑controlled ansatz that combines data‑encoding,
random layers, and entangling gates.  The observable set is chosen to
provide a rich basis for learning while keeping the number of measurements
moderate.  The public API matches the original build_classifier_circuit
signature so the same scripts can train either the classical or quantum
model interchangeably.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator

def generate_superposition_data(num_qubits: int, samples: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate data in the same superposition style as the classical counterpart.

    Returns features, classification labels, and regression targets.
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
    labels_class = (np.sin(thetas) > 0).astype(np.int64)
    labels_reg = np.sin(2 * thetas) * np.cos(phis)
    return states, labels_class, labels_reg

class SuperpositionDataset(torch.utils.data.Dataset):
    """Dataset returning quantum states and corresponding labels."""
    def __init__(self, samples: int, num_qubits: int, task: str = "classification"):
        self.states, self.labels_class, self.labels_reg = generate_superposition_data(num_qubits, samples)
        self.task = task

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, idx: int) -> dict:  # type: ignore[override]
        item = {"states": torch.tensor(self.states[idx], dtype=torch.cfloat)}
        if self.task == "classification":
            item["target"] = torch.tensor(self.labels_class[idx], dtype=torch.long)
        else:
            item["target"] = torch.tensor(self.labels_reg[idx], dtype=torch.float32)
        return item

def build_classifier_circuit(num_qubits: int,
                             depth: int = 2,
                             classification: bool = True) -> Tuple[QuantumCircuit, Iterable[ParameterVector], Iterable[ParameterVector], List[SparsePauliOp]]:
    """Construct a layered variational ansatz with data‑encoding and entanglement.

    Parameters
    ----------
    num_qubits
        Number of qubits / input features.
    depth
        Number of variational layers.
    classification
        When ``True`` the observable set targets a 2‑class decision surface;
        otherwise a single‑output regression observable is returned.

    Returns
    -------
    circuit
        QuantumCircuit ready for measurement.
    encoding
        ParameterVector for data‑encoding gates.
    weights
        ParameterVector for variational parameters.
    observables
        List of SparsePauliOp objects defining the measurement basis.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    qc = QuantumCircuit(num_qubits)

    # Data‑encoding layer (Rx for each feature)
    for qubit in range(num_qubits):
        qc.rx(encoding[qubit], qubit)

    # Variational layers
    idx = 0
    for _ in range(depth):
        # Single‑qubit rotations
        for qubit in range(num_qubits):
            qc.ry(weights[idx], qubit)
            idx += 1
        # Entangling pattern (CZ chain)
        for qubit in range(num_qubits - 1):
            qc.cz(qubit, qubit + 1)

    # Optional random layer (for richer feature space)
    if depth > 2:
        for _ in range(depth // 2):
            for qubit in range(num_qubits):
                qc.ry(np.pi / 4, qubit)
            for qubit in range(num_qubits - 1):
                qc.cz(qubit, qubit + 1)

    # Observables
    if classification:
        # Two Pauli‑Z observables targeting each qubit
        observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    else:
        # Single Pauli‑Z observable for regression
        observables = [SparsePauliOp.from_list([("Z" * num_qubits, 1)])]

    return qc, encoding, weights, observables

class HybridClassifier(EstimatorQNN):
    """Quantum variational classifier wrapping the circuit defined above.

    The estimator uses the StatevectorEstimator backend which is
    compatible with the circuit and observable set created by
    ``build_classifier_circuit``.  The class can be instantiated by
    passing the number of qubits and depth, mirroring the classical
    constructor signature for convenience.
    """
    def __init__(self, num_qubits: int, depth: int = 2):
        qc, encoding, weights, observables = build_classifier_circuit(num_qubits, depth, classification=True)
        estimator = StatevectorEstimator()
        super().__init__(
            circuit=qc,
            observables=observables,
            input_params=encoding,
            weight_params=weights,
            estimator=estimator,
        )
        self.num_qubits = num_qubits
        self.depth = depth

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run the circuit and return the measurement expectation values."""
        return super().forward(inputs)

__all__ = ["HybridClassifier", "build_classifier_circuit", "SuperpositionDataset"]
