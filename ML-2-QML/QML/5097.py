"""Quantum self‑attention that embeds the same parameters into a parametric circuit and uses a quantum estimator.

The quantum branch follows the structure of the original SelfAttention.py but augments it with:
- A parametric rotation layer for queries.
- A controlled‑phase entanglement block for keys.
- A measurement‑based estimator (EstimatorQNN) that outputs expectation values.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import Estimator
from qiskit import Aer

__all__ = ["SelfAttentionHybrid"]


class SelfAttentionHybrid:
    def __init__(self, n_qubits: int = 4, shots: int = 1024):
        """
        Parameters
        ----------
        n_qubits : int
            Number of qubits used to encode the attention parameters.
        shots : int
            Number of measurement shots for the estimator.
        """
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = Aer.get_backend("qasm_simulator")

        # Parameter placeholders
        self.rot_params = [Parameter(f"rot_{i}") for i in range(3 * n_qubits)]
        self.ent_params = [Parameter(f"ent_{i}") for i in range(n_qubits - 1)]

        # Build the base circuit
        self.circuit = QuantumCircuit(n_qubits)
        # Rotation layer (query)
        for i in range(n_qubits):
            self.circuit.rx(self.rot_params[3 * i], i)
            self.circuit.ry(self.rot_params[3 * i + 1], i)
            self.circuit.rz(self.rot_params[3 * i + 2], i)
        # Entanglement layer (key)
        for i in range(n_qubits - 1):
            self.circuit.cx(i, i + 1)
            self.circuit.rz(self.ent_params[i], i + 1)
        self.circuit.measure_all()

        # EstimatorQNN head – expectation of Z⊗…⊗Z
        observables = SparsePauliOp.from_list([("Z" * n_qubits, 1)])
        self.estimator = Estimator()
        self.estim_qnn = EstimatorQNN(
            circuit=self.circuit,
            observables=observables,
            input_params=self.rot_params[:n_qubits],   # first n_qubits as inputs
            weight_params=self.rot_params[n_qubits:], # remaining as weights
            estimator=self.estimator,
        )

    def run(
        self,
        backend,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ) -> np.ndarray:
        """
        Execute the quantum self‑attention circuit and return expectation values.

        Parameters
        ----------
        backend : qiskit.providers.Provider
            Quantum backend to execute the circuit.
        rotation_params : np.ndarray
            Shape ``(3 * n_qubits,)`` – rotation gate angles.
        entangle_params : np.ndarray
            Shape ``(n_qubits - 1,)`` – entanglement gate angles.
        shots : int
            Number of shots for the estimator.

        Returns
        -------
        np.ndarray
            Expectation value(s) returned by EstimatorQNN.
        """
        # Build parameter binding dictionary
        bind_dict = {
            param: val
            for param, val in zip(self.rot_params, rotation_params.tolist())
        }
        bind_dict.update(
            {param: val for param, val in zip(self.ent_params, entangle_params.tolist())}
        )
        # Evaluate with EstimatorQNN
        result = self.estim_qnn.evaluate([bind_dict], shots=shots)
        return result
