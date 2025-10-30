"""Hybrid quantum self‑attention with a quantum neural network estimator.

This module implements a parameterised quantum self‑attention circuit and
feeds its output into a Qiskit EstimatorQNN.  It merges the SelfAttention
quantum circuit design and the EstimatorQNN example into a single hybrid
estimator."""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, Aer, execute
from qiskit.circuit import Parameter
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import Estimator as StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp

class HybridSelfAttentionEstimator:
    """
    Quantum hybrid self‑attention + estimator.

    Parameters
    ----------
    n_qubits : int
        Number of qubits for the self‑attention circuit.
    shots : int
        Number of shots for the classical simulator.
    """
    def __init__(self, n_qubits: int = 4, shots: int = 1024):
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = Aer.get_backend("qasm_simulator")

        # Build attention and regression circuits
        self.attn_circ = self._build_attention_circuit()
        self.reg_circ = self._build_regression_circuit()

        # Full circuit: attention followed by regression
        self.full_circ = QuantumCircuit(self.n_qubits)
        self.full_circ.compose(self.attn_circ, inplace=True)
        self.full_circ.compose(self.reg_circ, inplace=True)

        # Observable for regression (expectation value of Z on all qubits)
        self.observable = SparsePauliOp.from_list([("Z" * self.n_qubits, 1)])

        # Prepare EstimatorQNN
        self.estimator = StatevectorEstimator()
        self.estimator_qnn = EstimatorQNN(
            circuit=self.full_circ,
            observables=self.observable,
            input_params=self._input_params(),
            weight_params=self._weight_params(),
            estimator=self.estimator,
        )

    def _build_attention_circuit(self) -> QuantumCircuit:
        """Parameterised self‑attention block."""
        qc = QuantumCircuit(self.n_qubits)
        self.rot_params = [Parameter(f"rot_{i}") for i in range(3 * self.n_qubits)]
        self.ent_params = [Parameter(f"ent_{i}") for i in range(self.n_qubits - 1)]

        # Rotations per qubit
        for i in range(self.n_qubits):
            qc.rx(self.rot_params[3 * i], i)
            qc.ry(self.rot_params[3 * i + 1], i)
            qc.rz(self.rot_params[3 * i + 2], i)

        # Entangling gates
        for i in range(self.n_qubits - 1):
            qc.crx(self.ent_params[i], i, i + 1)

        return qc

    def _build_regression_circuit(self) -> QuantumCircuit:
        """Simple variational circuit for regression."""
        qc = QuantumCircuit(self.n_qubits)
        self.reg_params = [Parameter(f"reg_{i}") for i in range(3 * self.n_qubits)]

        for i in range(self.n_qubits):
            qc.rx(self.reg_params[3 * i], i)
            qc.ry(self.reg_params[3 * i + 1], i)
            qc.rz(self.reg_params[3 * i + 2], i)

        return qc

    def _input_params(self):
        """Return the list of input parameters (classical inputs)."""
        # For illustration we expose the first rotation parameter as input
        return [self.rot_params[0]]

    def _weight_params(self):
        """Return the list of weight parameters (quantum trainable params)."""
        return self.reg_params

    def run(self, input_value: float) -> dict:
        """
        Execute the hybrid circuit.

        Parameters
        ----------
        input_value : float
            Value to bind to the single input parameter.

        Returns
        -------
        dict
            Mapping from observable names to expectation values.
        """
        feed_dict = {self.rot_params[0]: input_value}
        result = self.estimator_qnn.predict(feed_dict)
        return result

__all__ = ["HybridSelfAttentionEstimator"]
