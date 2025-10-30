"""Quantum implementation of kernel evaluation and variational classification.

The class `QuantumKernelMethod` mirrors the public API of the classical
counterpart but replaces the RBF kernel with a parameterised quantum
kernel built from a feature‑map (RX encoding) and a variational ansatz
(RY + CZ).  Training is performed with Qiskit’s `VQC` algorithm, which
optimises the ansatz parameters via a gradient‑based optimiser.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
from qiskit import QuantumCircuit, Aer
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.utils import QuantumInstance
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.kernels import QuantumKernel


class QuantumKernelMethod:
    """Quantum kernel evaluation and variational classifier.

    Parameters
    ----------
    num_qubits : int
        Number of qubits / feature dimensionality.
    depth : int, default 3
        Depth of the variational ansatz.
    backend : str, default "aer_simulator_statevector"
        Qiskit backend used for simulation.
    optimizer : str, default "COBYLA"
        Optimiser passed to Qiskit’s VQC.
    epochs : int, default 100
        Training epochs for the VQC.
    """

    def __init__(
        self,
        num_qubits: int,
        depth: int = 3,
        backend: str = "aer_simulator_statevector",
        optimizer: str = "COBYLA",
        epochs: int = 100,
    ) -> None:
        self.num_qubits = num_qubits
        self.depth = depth

        # Parameter vectors for encoding and variational layers
        self.encoding = ParameterVector("x", num_qubits)
        self.weights = ParameterVector("theta", num_qubits * depth)

        # Feature‑map circuit (data encoding)
        self.feature_map = QuantumCircuit(num_qubits)
        for qubit, param in enumerate(self.encoding):
            self.feature_map.rx(param, qubit)

        # Variational ansatz
        self.ansatz = QuantumCircuit(num_qubits)
        idx = 0
        for _ in range(depth):
            for qubit in range(num_qubits):
                self.ansatz.ry(self.weights[idx], qubit)
                idx += 1
            for qubit in range(num_qubits - 1):
                self.ansatz.cz(qubit, qubit + 1)

        # Observables: Z on each qubit
        self.observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
            for i in range(num_qubits)
        ]

        # Quantum instance for simulation
        self.qiskit_backend = Aer.get_backend(backend)
        self.quantum_instance = QuantumInstance(self.qiskit_backend, shots=8192)

        # Quantum kernel
        self.kernel = QuantumKernel(
            feature_map=self.feature_map,
            ansatz=self.ansatz,
            backend=self.qiskit_backend,
        )

        # Variational quantum classifier
        self.model = VQC(
            feature_map=self.feature_map,
            ansatz=self.ansatz,
            optimizer=optimizer,
            quantum_instance=self.quantum_instance,
            epochs=epochs,
            verbose=0,
        )

    # ------------------------------------------------------------------
    # Kernel utilities
    # ------------------------------------------------------------------
    def kernel_matrix(self, X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
        """Return the quantum kernel matrix between X and Y."""
        Y = X if Y is None else Y
        return np.array(self.kernel.evaluate(X, Y))

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the variational quantum classifier."""
        self.model.fit(X, y)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class predictions for X."""
        return self.model.predict(X)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return the accuracy of the fitted model."""
        preds = self.predict(X)
        return np.mean(preds == y)


__all__ = ["QuantumKernelMethod"]
