"""SelfAttentionHybrid: quantum implementation with Pauli‑Z feature extraction.

The quantum version mirrors the classical API but uses a variational
circuit to produce a feature vector that can be passed to a classical head
for regression or classification.  The circuit is built with Qiskit; the
measurements are converted to expectation values of Pauli‑Z on each qubit.
The module can be dropped into a pipeline that already expects a
`SelfAttention`‑style object.\n
Author: GPT‑OSS‑20B
"""

from __future__ import annotations

import math
from typing import Literal, Tuple, Iterable

import numpy as np
import torch
import torch.nn as nn
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

# --------------------------------------------------------------------------- #
# 1. Helper: build a small variational circuit that emulates attention       #
# --------------------------------------------------------------------------- #

def _build_attention_circuit(
    n_qubits: int,
    rotation_params: np.ndarray,
    entangle_params: np.ndarray,
) -> QuantumCircuit:
    """Internal helper that constructs the attention circuit.

    The circuit applies a rotation on each qubit followed by a nearest‑neighbour
    CRX gate sequence.  This is a lightweight analogue of the classical
    self‑attention weight matrices.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    rotation_params : np.ndarray shape (3*n_qubits,)
        Parameters for Rx, Ry, Rz on each qubit.
    entangle_params : np.ndarray shape (n_qubits - 1,)
        Parameters for CRX entangling gates.

    Returns
    -------
    QuantumCircuit
    """
    qr = QuantumRegister(n_qubits, "q")
    cr = ClassicalRegister(n_qubits, "c")
    circuit = QuantumCircuit(qr, cr)

    # Rotation layer
    for i in range(n_qubits):
        circuit.rx(rotation_params[3 * i], i)
        circuit.ry(rotation_params[3 * i + 1], i)
        circuit.rz(rotation_params[3 * i + 2], i)

    # Entangling layer
    for i in range(n_qubits - 1):
        circuit.crx(entangle_params[i], i, i + 1)

    circuit.measure(qr, cr)
    return circuit


# --------------------------------------------------------------------------- #
# 2. Quantum self‑attention hybrid class                                     #
# --------------------------------------------------------------------------- #

class QuantumSelfAttentionHybrid:
    """Quantum version of the SelfAttentionHybrid.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the circuit (also the dimensionality of the
        returned feature vector).
    task : Literal['regression', 'classification']
        Which downstream head to attach.
    device : torch.device, optional
        Device for the classical head (default: cpu).
    """

    def __init__(
        self,
        n_qubits: int,
        task: Literal["regression", "classification"],
        device: torch.device | None = None,
    ):
        self.n_qubits = n_qubits
        self.task = task
        self.device = device or torch.device("cpu")

        # Classical head to interpret measurement outcomes
        if task == "regression":
            self.head = nn.Sequential(
                nn.Linear(n_qubits, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            ).to(self.device)
        elif task == "classification":
            self.head = nn.Sequential(
                nn.Linear(n_qubits, 32),
                nn.ReLU(),
                nn.Linear(32, 2),
            ).to(self.device)
        else:
            raise ValueError("task must be'regression' or 'classification'")

        # Backend for simulation
        self.backend = Aer.get_backend("qasm_simulator")

    def _counts_to_expectation(self, counts: dict[str, int]) -> torch.Tensor:
        """Convert measurement counts to Pauli‑Z expectation values."""
        total = sum(counts.values())
        expectations = torch.zeros(self.n_qubits, device=self.device, dtype=torch.float32)

        for bitstring, freq in counts.items():
            bits = np.array([int(b) for b in bitstring[::-1]])  # reversed order
            z = 1 - 2 * bits  # map 0->+1, 1->-1
            expectations += torch.tensor(z, device=self.device, dtype=torch.float32) * freq / total

        return expectations

    def forward(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ) -> torch.Tensor:
        """Run the circuit and apply the head.

        Parameters
        ----------
        rotation_params : np.ndarray shape (3*n_qubits,)
        entangle_params : np.ndarray shape (n_qubits-1,)
        shots : int, optional
            Number of shots for the simulator.
        """
        circuit = _build_attention_circuit(
            self.n_qubits, rotation_params, entangle_params
        )
        job = execute(circuit, self.backend, shots=shots)
        result = job.result()
        counts = result.get_counts(circuit)

        features = self._counts_to_expectation(counts)
        return self.head(features)


# --------------------------------------------------------------------------- #
# 3. Factory helpers (mirroring the classical side)                         #
# --------------------------------------------------------------------------- #

def build_classifier_circuit(
    num_qubits: int,
    depth: int,
) -> Tuple[QuantumCircuit, Iterable, Iterable, list[SparsePauliOp]]:
    """Return a simple layered ansatz and metadata.

    This function is identical to the quantum reference but is kept in the
    same module to provide a unified API.

    Returns
    -------
    circuit : QuantumCircuit
    encoding : list[Parameter]
    weights : list[Parameter]
    observables : list[SparsePauliOp]
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    for i, param in enumerate(encoding):
        circuit.rx(param, i)

    idx = 0
    for _ in range(depth):
        for i in range(num_qubits):
            circuit.ry(weights[idx], i)
            idx += 1
        for i in range(num_qubits - 1):
            circuit.cz(i, i + 1)

    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]
    return circuit, list(encoding), list(weights), observables


__all__ = [
    "QuantumSelfAttentionHybrid",
    "build_classifier_circuit",
]
