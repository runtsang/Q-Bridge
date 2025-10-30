"""Quantum self‑attention with a Qiskit EstimatorQNN head, inspired by SelfAttention.py and EstimatorQNN.py."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator as StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN


class SelfAttentionEstimator:
    """
    Quantum‑classical hybrid model that mirrors the classical SelfAttentionEstimator.
    A parameterised quantum circuit implements the self‑attention block; the
    expectation values of Y on each qubit are fed into a small classical MLP
    to produce a regression output.
    """

    def __init__(self, n_qubits: int = 4, hidden_dim: int = 8) -> None:
        self.n_qubits = n_qubits
        self.hidden_dim = hidden_dim

        # --- Quantum circuit template ---
        self.qc = QuantumCircuit(n_qubits)
        self.input_params = [Parameter(f"x{i}") for i in range(n_qubits * 3)]
        self.weight_params = [Parameter(f"w{i}") for i in range(n_qubits - 1)]

        for i in range(n_qubits):
            self.qc.rx(self.input_params[3 * i], i)
            self.qc.ry(self.input_params[3 * i + 1], i)
            self.qc.rz(self.input_params[3 * i + 2], i)

        for i in range(n_qubits - 1):
            self.qc.crx(self.weight_params[i], i, i + 1)

        # Observables: Y on each qubit
        pauli_list = []
        for i in range(n_qubits):
            pauli_str = "I" * i + "Y" + "I" * (n_qubits - i - 1)
            pauli_list.append((pauli_str, 1.0))
        self.observables = SparsePauliOp.from_list(pauli_list)

        # Estimator (state‑vector backend)
        self.estimator = StatevectorEstimator()
        self.qnn = EstimatorQNN(
            circuit=self.qc,
            observables=self.observables,
            input_params=self.input_params,
            weight_params=self.weight_params,
            estimator=self.estimator,
        )

        # Classical regression head
        self.regressor = nn.Sequential(
            nn.Linear(n_qubits, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def run(
        self,
        backend,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ) -> np.ndarray:
        """
        Execute the hybrid model.

        Parameters
        ----------
        backend
            Qiskit backend to run the circuit (e.g., Aer.get_backend('statevector_simulator')).
        rotation_params : np.ndarray
            Parameters for the rotation gates (shape: (n_qubits * 3,)).
        entangle_params : np.ndarray
            Parameters for the entangling CRX gates (shape: (n_qubits - 1,)).
        shots : int, optional
            Number of shots for the estimator.

        Returns
        -------
        np.ndarray
            Regression output (shape: (1,)).
        """
        param_dict = {
            param: val for param, val in zip(self.input_params, rotation_params)
        }
        param_dict.update(
            {param: val for param, val in zip(self.weight_params, entangle_params)}
        )

        # Predict expectation values
        exp_vals = self.qnn.predict(backend, param_dict, shots=shots)

        # The EstimatorQNN returns a list of expectation values, one per observable.
        exp_tensor = torch.as_tensor(exp_vals, dtype=torch.float32)
        out = self.regressor(exp_tensor)
        return out.detach().numpy()


__all__ = ["SelfAttentionEstimator"]
