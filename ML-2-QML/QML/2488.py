"""Hybrid quantum kernel estimator using Qiskit EstimatorQNN."""
from __future__ import annotations

import numpy as np
from qiskit.circuit import Parameter
from qiskit import QuantumCircuit, Aer
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator
from typing import Sequence

# --------------------------------------------------------------------------- #
# Quantum kernel primitives – kept for backward compatibility
# --------------------------------------------------------------------------- #
def _encode_qubits(x: np.ndarray, n_qubits: int) -> QuantumCircuit:
    """Encode a classical vector into a quantum state via Ry rotations."""
    qc = QuantumCircuit(n_qubits)
    for i, val in enumerate(x[:n_qubits]):
        qc.ry(val, i)
    return qc

def quantum_kernel_matrix(a: Sequence[np.ndarray], b: Sequence[np.ndarray],
                          n_qubits: int = 4) -> np.ndarray:
    """Compute Gram matrix via state‑vector overlaps."""
    mat = np.zeros((len(a), len(b)), dtype=np.float64)
    for i, x in enumerate(a):
        qc_x = _encode_qubits(x, n_qubits)
        sv_x = Statevector.from_instruction(qc_x)
        for j, y in enumerate(b):
            qc_y = _encode_qubits(y, n_qubits)
            sv_y = Statevector.from_instruction(qc_y)
            mat[i, j] = np.abs(np.vdot(sv_x.data, sv_y.data)) ** 2
    return mat

# --------------------------------------------------------------------------- #
# Hybrid estimator – quantum side
# --------------------------------------------------------------------------- #
class HybridQuantumKernelEstimator:
    """Quantum kernel + Qiskit EstimatorQNN for regression."""
    def __init__(self, n_qubits: int = 4, lr: float = 0.01, epochs: int = 200) -> None:
        self.n_qubits = n_qubits
        self.lr = lr
        self.epochs = epochs
        self.estimator_qnn: EstimatorQNN | None = None
        self.train_X: np.ndarray | None = None

    def _build_circuit(self, input_params: Sequence[Parameter],
                       weight_params: Sequence[Parameter]) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        # Input encoding
        for idx, param in enumerate(input_params):
            qc.ry(param, idx)
        # Weight encoding
        for idx, param in enumerate(weight_params):
            qc.rx(param, idx)
        return qc

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the EstimatorQNN on the provided data."""
        input_params = [Parameter(f"input{i}") for i in range(self.n_qubits)]
        weight_params = [Parameter(f"weight{i}") for i in range(self.n_qubits)]
        qc = self._build_circuit(input_params, weight_params)

        # Observable: Pauli‑Y on the first qubit
        observable = SparsePauliOp.from_list([("Y" + "I" * (self.n_qubits - 1), 1)])

        # Use the state‑vector simulator as backend
        backend = Aer.get_backend("statevector_simulator")
        estimator = StatevectorEstimator(backend=backend)

        self.estimator_qnn = EstimatorQNN(
            circuit=qc,
            observables=observable,
            input_params=input_params,
            weight_params=weight_params,
            estimator=estimator,
        )
        # Train with the built‑in optimizer
        self.estimator_qnn.fit(X, y, epochs=self.epochs, lr=self.lr)
        self.train_X = X

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the trained EstimatorQNN."""
        if self.estimator_qnn is None:
            raise RuntimeError("EstimatorQNN has not been trained.")
        preds = self.estimator_qnn.predict(X)
        return np.array(preds).reshape(-1)

    def kernel_matrix(self, a: Sequence[np.ndarray], b: Sequence[np.ndarray]) -> np.ndarray:
        """Return the quantum kernel Gram matrix."""
        return quantum_kernel_matrix(a, b, self.n_qubits)

__all__ = ["quantum_kernel_matrix", "HybridQuantumKernelEstimator"]
