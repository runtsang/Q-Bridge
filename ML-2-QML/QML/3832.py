"""Hybrid kernel ridge regressor using a Qiskit quantum kernel and a variational EstimatorQNN.

The class :class:`HybridQuantumKernelRegressor` implements the same
interface as the classical version but relies on Qiskit for the quantum
kernel evaluation.  The quantum kernel is built from a single‑qubit
Ry encoding, and the EstimatorQNN is a single‑qubit variational
circuit whose expectation of the Y Pauli operator is used as the
regression output.  The class can be trained end‑to‑end with gradient
descent on the circuit parameters via the ``StatevectorEstimator``.
"""

import numpy as np
import torch
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN

# --------------------------------------------------------------------------- #
# Quantum kernel built from an analytic single‑qubit Ry encoding
# --------------------------------------------------------------------------- #
class QuantumKernel:
    """
    Computes the overlap kernel
    K(x, y) = |<φ(x)|φ(y)>|^2
    where |φ(z)> = ⊗_i Ry(z_i) |0> for a single qubit.
    For a single qubit the closed form is
        cos((x - y)/2)^2
    For multiple qubits the kernel is the product of the single‑qubit
    kernels.
    """

    def __init__(self, n_qubits: int = 1) -> None:
        self.n_qubits = n_qubits

    def _single_qubit_kernel(self, x: np.ndarray, y: np.ndarray) -> float:
        return np.cos((x[0] - y[0]) / 2.0) ** 2

    def __call__(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Return the Gram matrix between ``X`` and ``Y``."""
        K = np.zeros((X.shape[0], Y.shape[0]))
        for i, x in enumerate(X):
            for j, y in enumerate(Y):
                prod = 1.0
                for k in range(self.n_qubits):
                    prod *= np.cos((x[k] - y[k]) / 2.0) ** 2
                K[i, j] = prod
        return K

# --------------------------------------------------------------------------- #
# Variational EstimatorQNN circuit
# --------------------------------------------------------------------------- #
def create_estimator_qnn(n_qubits: int = 1) -> QiskitEstimatorQNN:
    """
    Builds a Qiskit EstimatorQNN that uses a single‑qubit Ry (input)
    and Rx (weight) gate, measuring the Y Pauli operator.
    """
    input_param = Parameter("input")
    weight_param = Parameter("weight")

    qc = QuantumCircuit(n_qubits)
    qc.h(0)
    qc.ry(input_param, 0)
    qc.rx(weight_param, 0)

    observable = SparsePauliOp.from_list([("Y" * n_qubits, 1)])

    estimator = StatevectorEstimator()

    return QiskitEstimatorQNN(
        circuit=qc,
        observables=observable,
        input_params=[input_param],
        weight_params=[weight_param],
        estimator=estimator,
    )

# --------------------------------------------------------------------------- #
# Hybrid kernel ridge regressor using the Qiskit kernel
# --------------------------------------------------------------------------- #
class HybridQuantumKernelRegressor:
    """
    Kernel ridge regressor that uses the analytic Qiskit quantum kernel.
    The interface mirrors the classical version for consistency.
    """

    def __init__(self, n_qubits: int = 1, reg: float = 1e-3) -> None:
        self.kernel = QuantumKernel(n_qubits)
        self.reg = reg
        self.X_train: np.ndarray | None = None
        self.alpha: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Compute the hybrid kernel matrix and solve for the coefficients.
        """
        self.X_train = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1, 1)

        K = self.kernel(self.X_train, self.X_train)
        I = self.reg * np.eye(K.shape[0])
        self.alpha = np.linalg.solve(K + I, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict on new data.
        """
        if self.X_train is None or self.alpha is None:
            raise RuntimeError("Model has not been fitted.")
        K_test = self.kernel(np.asarray(X, dtype=np.float64), self.X_train)
        return (K_test @ self.alpha).ravel()

    # --------------------------------------------------------------------- #
    # EstimatorQNN helper
    # --------------------------------------------------------------------- #
    def estimator_qnn(self, n_qubits: int = 1) -> QiskitEstimatorQNN:
        """Return a Qiskit EstimatorQNN instance for the given number of qubits."""
        return create_estimator_qnn(n_qubits)

__all__ = [
    "QuantumKernel",
    "create_estimator_qnn",
    "HybridQuantumKernelRegressor",
]
