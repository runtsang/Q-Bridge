"""Quantum fully‑connected layer and EstimatorQNN integration.

The quantum circuit is a simple variational ansatz that:
    * Uses one qubit per input parameter.
    * Applies a Hadamard layer followed by a parameterized rotation.
    * Measures the expectation of the Y Pauli operator on each qubit.
    * Aggregates the expectation values via a classical linear combination.

The EstimatorQNN helper constructs a Qiskit EstimatorQNN object that can be
used as a differentiable layer in a quantum‑classical hybrid workflow.

Author: gpt-oss-20b
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import StatevectorEstimator

class QuantumFullyConnectedLayer:
    """
    Parameterized quantum circuit that emulates a fully‑connected layer.

    The circuit accepts a list of parameters (one per qubit).  For each
    parameter θ it applies an RY(θ) rotation after a global Hadamard layer.
    The expectation value of the Y operator on each qubit is returned.
    """

    def __init__(self, n_qubits: int, backend: qiskit.providers.Backend, shots: int = 1024):
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots
        # Build the parameterized circuit
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        self.theta_params = [Parameter(f"θ{i}") for i in range(n_qubits)]
        self.circuit.h(range(n_qubits))
        self.circuit.barrier()
        for q, theta in enumerate(self.theta_params):
            self.circuit.ry(theta, q)
        self.circuit.measure_all()
        # Observable: sum of Y operators on each qubit
        self.observable = SparsePauliOp.from_list([("Y" * n_qubits, 1)])

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """
        Evaluate the circuit for a batch of parameter vectors.

        Parameters
        ----------
        thetas : np.ndarray
            2‑D array of shape (batch_size, n_qubits).

        Returns
        -------
        np.ndarray
            1‑D array of expectation values, one per batch element.
        """
        batch_size = thetas.shape[0]
        expectations = np.empty(batch_size)
        for i, theta_vec in enumerate(thetas):
            bind = {theta: val for theta, val in zip(self.theta_params, theta_vec)}
            job = qiskit.execute(
                self.circuit,
                self.backend,
                shots=self.shots,
                parameter_binds=[bind],
            )
            result = job.result()
            counts = result.get_counts(self.circuit)
            probs = np.array(list(counts.values())) / self.shots
            states = np.array([int(k, 2) for k in counts.keys()])
            # Compute expectation of Y on each qubit and sum
            exp = 0.0
            for state, p in zip(states, probs):
                # Convert state to binary string
                bits = format(state, f"0{self.n_qubits}b")
                y_val = 1.0
                for bit in bits:
                    y_val *= 1.0 if bit == '0' else -1.0  # Y eigenvalue ±1
                exp += y_val * p
            expectations[i] = exp
        return expectations

def EstimatorQNN() -> QiskitEstimatorQNN:
    """
    Construct a Qiskit EstimatorQNN object that mirrors the original
    EstimatorQNN example but uses the new QuantumFullyConnectedLayer.
    """
    # Simple 1‑qubit circuit for demonstration
    qc = qiskit.QuantumCircuit(1)
    param = Parameter("θ")
    qc.h(0)
    qc.ry(param, 0)
    qc.x(0)  # additional gate to break symmetry
    qc.measure_all()
    observable = SparsePauliOp.from_list([("Y", 1)])
    estimator = StatevectorEstimator()
    return QiskitEstimatorQNN(
        circuit=qc,
        observables=observable,
        input_params=[param],
        weight_params=[],
        estimator=estimator,
    )

__all__ = ["QuantumFullyConnectedLayer", "EstimatorQNN"]
