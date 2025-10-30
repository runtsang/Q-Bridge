"""Quantum hybrid kernel using Qiskit.

The HybridKernel class builds a parameterized circuit that encodes two
classical data points and evaluates the fidelity between the resulting
statevectors.  The circuit structure is inspired by the SamplerQNN
example: it uses Ry rotations for data encoding, a CX entanglement
layer, and additional Ry gates for trainable weights.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from typing import Sequence, Tuple

class HybridKernel:
    """
    Quantum kernel implemented as a fidelity of statevectors from a
    parameterized circuit.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the circuit.
    seed : int | None, optional
        Random seed for initializing trainable weights.
    """

    def __init__(self, n_qubits: int = 2, seed: int | None = None) -> None:
        self.n_qubits = n_qubits
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        # initialize trainable weights for each qubit
        self.weights = self.rng.normal(size=n_qubits).astype(np.float64)

        # simulator for statevector extraction
        self.sim = AerSimulator(method="statevector")

    def _build_circuit(self, data: np.ndarray, weights: np.ndarray | None = None) -> QuantumCircuit:
        """Return a circuit that encodes 'data' into the state."""
        if weights is None:
            weights = self.weights
        qc = QuantumCircuit(self.n_qubits)
        # Data encoding with Ry rotations
        for i in range(self.n_qubits):
            qc.ry(data[i], i)
        # Entanglement
        qc.cx(0, 1)
        # Trainable Ry rotations
        for i in range(self.n_qubits):
            qc.ry(weights[i], i)
        return qc

    def _statevector_from_data(self, data: Sequence[float]) -> Statevector:
        """Compute the statevector for a given data point."""
        qc = self._build_circuit(np.array(data, dtype=np.float64))
        result = self.sim.run(qc).result()
        return Statevector.from_int(result.get_statevector(qc), dims=(2,) * self.n_qubits)

    def __call__(self, x: Sequence[float], y: Sequence[float]) -> float:
        """Compute kernel value between two data points."""
        psi_x = self._statevector_from_data(x)
        psi_y = self._statevector_from_data(y)
        overlap = np.abs(np.vdot(psi_x.data, psi_y.data)) ** 2
        return float(overlap)

    def kernel_matrix(self, a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> np.ndarray:
        """Return Gram matrix between two sets of data."""
        return np.array([[self(a_i, b_j) for b_j in b] for a_i in a])

    def set_weights(self, weights: Sequence[float]) -> None:
        """Externally set trainable weights."""
        if len(weights)!= self.n_qubits:
            raise ValueError("Weight vector length must match number of qubits.")
        self.weights = np.asarray(weights, dtype=np.float64)

    def reset_weights(self) -> None:
        """Reinitialize weights randomly."""
        self.weights = self.rng.normal(size=self.n_qubits).astype(np.float64)
