"""Quantum neural network estimator with a two‑qubit entangled variational ansatz."""
from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import Estimator as StatevectorEstimator
import numpy as np
from typing import Tuple

class EstimatorQNN:
    """Quantum regression network with an entangled variational ansatz."""
    def __init__(self, *, device: str = "aer_simulator.statevector") -> None:
        self.device = device
        self.circuit = self._build_ansatz()
        self.observables = self._build_observable()
        self.estimator = StatevectorEstimator(backend=self.device)
        self.qnn = QiskitEstimatorQNN(
            circuit=self.circuit,
            observables=self.observables,
            input_params=[self.circuit.parameters[0]],
            weight_params=[self.circuit.parameters[1]],
            estimator=self.estimator,
        )
        self.is_trained = False

    def _build_ansatz(self) -> QuantumCircuit:
        theta, phi = Parameter("θ"), Parameter("φ")
        qc = QuantumCircuit(2)
        qc.ry(theta, 0)
        qc.rx(phi, 1)
        qc.cx(0, 1)
        qc.cx(1, 0)
        qc.ry(theta, 1)
        qc.rx(phi, 0)
        return qc

    def _build_observable(self) -> SparsePauliOp:
        pauli = Pauli("YY")
        return SparsePauliOp(pauli)

    def fit(self, X: np.ndarray, y: np.ndarray, *, max_iter: int = 200, tol: float = 1e-4) -> None:
        self.qnn.fit(
            X=X.reshape(-1, 1),
            y=y,
            method="gradient",
            max_iter=max_iter,
            tol=tol,
        )
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("EstimatorQNN has not been trained yet.")
        return self.qnn.predict(X.reshape(-1, 1))

__all__ = ["EstimatorQNN"]
