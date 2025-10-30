"""Hybrid quantum classifier combining a data‑encoding quanvolution layer with a layered variational ansatz.

The circuit first encodes the input data into rotation angles (the “quanvolution” stage) using a
threshold‑based mapping.  It then applies a depth‑wise variational ansatz consisting of
single‑qubit rotations followed by a CZ entangling pattern, mirroring the classic feed‑forward
architecture.  The public ``build_classifier_circuit`` function returns the circuit, encoding
parameters, variational weights, and observables in a format compatible with the original
QuantumClassifierModel.py.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List
import numpy as np
from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit import ParameterVector, Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.random import random_circuit

# --------------------------------------------------------------------------- #
# QuanvCircuit – data‑encoding subcircuit (quanvolution)
# --------------------------------------------------------------------------- #
class QuanvCircuit:
    """Data‑encoding subcircuit that applies a threshold‑based RX rotation to each qubit."""
    def __init__(self, kernel_size: int, backend, shots: int = 100, threshold: float = 127.0) -> None:
        self.n_qubits = kernel_size ** 2
        self.backend = backend
        self.shots = shots
        self.threshold = threshold
        self.circuit = QuantumCircuit(self.n_qubits)
        # Parameterised rotation for data encoding
        self.theta = [Parameter(f"x_{i}") for i in range(self.n_qubits)]
        for i, param in enumerate(self.theta):
            self.circuit.rx(param, i)
        self.circuit.barrier()
        # Entangling pattern (random 2‑gate layer for diversity)
        self.circuit += random_circuit(self.n_qubits, 2)
        self.circuit.measure_all()

    def run(self, data: np.ndarray) -> float:
        """
        Execute the circuit on classical data.

        Parameters
        ----------
        data : np.ndarray
            2‑D array of shape ``(kernel_size, kernel_size)``.
        Returns
        -------
        float
            Average probability of measuring |1> across all qubits.
        """
        data_flat = np.reshape(data, (self.n_qubits,))
        bind = {param: np.pi if val > self.threshold else 0 for param, val in zip(self.theta, data_flat)}
        job = execute(self.circuit, self.backend,
                      shots=self.shots, parameter_binds=[bind])
        result = job.result().get_counts(self.circuit)
        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val
        return counts / (self.shots * self.n_qubits)

# --------------------------------------------------------------------------- #
# build_classifier_circuit – public API
# --------------------------------------------------------------------------- #
def build_classifier_circuit(num_qubits: int, depth: int,
                             conv_kernel: int = 2, conv_threshold: float = 127.0
                            ) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """
    Construct a hybrid quantum classifier circuit.

    Parameters
    ----------
    num_qubits : int
        Number of qubits (typically ``conv_kernel**2`` for a square patch).
    depth : int
        Number of variational layers.
    conv_kernel : int, optional
        Size of the square patch for the quanvolution stage. Defaults to 2.
    conv_threshold : float, optional
        Threshold used for data encoding. Defaults to 127.0.

    Returns
    -------
    Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]
        The circuit, encoding parameters, variational weights, and observables.
    """
    # Ensure the circuit size matches the kernel
    if num_qubits!= conv_kernel ** 2:
        raise ValueError("num_qubits must equal conv_kernel**2 for the quanvolution stage.")
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    # Quanvolution encoding
    for q, param in enumerate(encoding):
        circuit.rx(param, q)

    # Variational ansatz
    idx = 0
    for _ in range(depth):
        for q in range(num_qubits):
            circuit.ry(weights[idx], q)
            idx += 1
        for q in range(num_qubits - 1):
            circuit.cz(q, q + 1)

    # Observables – single‑qubit Z measurements
    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
                   for i in range(num_qubits)]
    return circuit, list(encoding), list(weights), observables

__all__ = ["build_classifier_circuit"]
