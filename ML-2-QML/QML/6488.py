"""
qml_self_attention_hybrid.py

Quantum self‑attention block followed by a Qiskit EstimatorQNN.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import Estimator as StatevectorEstimator


class QuantumSelfAttentionHybrid:
    """Quantum self‑attention module with a variational estimator.

    Parameters
    ----------
    n_qubits : int
        Number of qubits used for the self‑attention circuit.
    observable : SparsePauliOp, optional
        Observable to evaluate; defaults to a single‑qubit Y operator.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        observable: SparsePauliOp | None = None,
    ) -> None:
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")
        self.observable = (
            observable
            if observable is not None
            else SparsePauliOp.from_list([("Y" * n_qubits, 1)])
        )
        self.estimator = StatevectorEstimator()
        self._build_estimator_qnn()

    def _rotation_params(self, params: np.ndarray) -> list[Parameter]:
        """Return a list of rotation parameters for the circuit."""
        return [Parameter(f"rot_{i}") for i in range(self.n_qubits * 3)]

    def _entangle_params(self, params: np.ndarray) -> list[Parameter]:
        """Return a list of entanglement parameters for the circuit."""
        return [Parameter(f"ent_{i}") for i in range(self.n_qubits - 1)]

    def _build_circuit(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
    ) -> QuantumCircuit:
        """Construct the self‑attention variational circuit."""
        circ = QuantumCircuit(self.qr, self.cr)
        # Rotation layer
        for i in range(self.n_qubits):
            circ.rx(rotation_params[3 * i], i)
            circ.ry(rotation_params[3 * i + 1], i)
            circ.rz(rotation_params[3 * i + 2], i)
        # Entanglement layer
        for i in range(self.n_qubits - 1):
            circ.cx(i, i + 1)
            circ.rz(entangle_params[i], i + 1)
        circ.measure(self.qr, self.cr)
        return circ

    def _build_estimator_qnn(self) -> None:
        """Wrap the circuit in a Qiskit EstimatorQNN."""
        # Use placeholder parameters; actual values supplied at run time
        rotation_params = np.zeros(self.n_qubits * 3)
        entangle_params = np.zeros(self.n_qubits - 1)
        qc = self._build_circuit(rotation_params, entangle_params)

        self.estimator_qnn = EstimatorQNN(
            circuit=qc,
            observables=self.observable,
            input_params=[],
            weight_params=[],
            estimator=self.estimator,
        )

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ) -> dict:
        """
        Execute the quantum self‑attention circuit and return measurement counts.

        Parameters
        ----------
        rotation_params : np.ndarray
            Array of shape (n_qubits * 3,) defining rotation angles.
        entangle_params : np.ndarray
            Array of shape (n_qubits - 1,) defining entanglement angles.
        shots : int, optional
            Number of shots for the simulator.

        Returns
        -------
        dict
            Measurement outcome counts from the simulator.
        """
        # Rebuild circuit with actual parameters
        circuit = self._build_circuit(rotation_params, entangle_params)
        backend = qiskit.Aer.get_backend("qasm_simulator")
        job = qiskit.execute(circuit, backend, shots=shots)
        return job.result().get_counts(circuit)


__all__ = ["QuantumSelfAttentionHybrid"]
