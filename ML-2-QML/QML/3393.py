"""Quantum self‑attention estimator that mirrors the hybrid architecture.

This module builds a variational circuit that implements a simple self‑attention
mechanism.  Rotation parameters are supplied by the data, while entanglement‑like
parameters are treated as trainable weights.  The circuit is wrapped in
Qiskit‑Machine‑Learning's EstimatorQNN so that it can be plugged into a larger
quantum‑classical workflow.

Scaling paradigm: `combination` – the quantum layer is a feature extractor that
feeds into a downstream classical model.
"""

from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import Estimator
import numpy as np


class HybridEstimatorQNN:
    """
    Quantum self‑attention estimator.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the self‑attention block (default 4).
    """

    def __init__(self, n_qubits: int = 4) -> None:
        self.n_qubits = n_qubits
        self.rotation_params = [Parameter(f"rot_{i}") for i in range(n_qubits)]
        self.entangle_params = [Parameter(f"ent_{i}") for i in range(n_qubits - 1)]
        self.circuit = self._build_circuit()

        # Pauli‑Y observable on the last qubit
        self.observable = SparsePauliOp.from_list([("Y" + "I" * (n_qubits - 1), 1)])

        # Statevector estimator for expectation values
        self.estimator = Estimator()
        self.qnn = EstimatorQNN(
            circuit=self.circuit,
            observables=self.observable,
            input_params=self.rotation_params,
            weight_params=self.entangle_params,
            estimator=self.estimator,
        )

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        # Rotation layer
        for i, param in enumerate(self.rotation_params):
            qc.rx(param, i)
            qc.ry(param, i)
            qc.rz(param, i)
        # Entanglement layer (CRX gates)
        for i, ent in enumerate(self.entangle_params):
            qc.crx(ent, i, i + 1)
        return qc

    def run(
        self,
        input_values: np.ndarray,
        weight_values: np.ndarray,
        backend_name: str = "qasm_simulator",
        shots: int = 1024,
    ) -> dict[str, int]:
        """
        Execute the self‑attention circuit on a back‑end.

        Parameters
        ----------
        input_values : np.ndarray
            Array of rotation parameters (shape (n_qubits,)).
        weight_values : np.ndarray
            Array of entanglement parameters (shape (n_qubits-1,)).
        backend_name : str, optional
            Name of the Qiskit simulator to run on.
        shots : int, optional
            Number of shots for the qasm simulator.
        """
        param_dict = dict(zip(self.rotation_params, input_values))
        param_dict.update(dict(zip(self.entangle_params, weight_values)))

        qc = self.circuit.assign_parameters(param_dict)
        backend = Aer.get_backend(backend_name)
        job = execute(qc, backend, shots=shots)
        result = job.result()
        return result.get_counts(qc)

    def expectation(
        self,
        input_values: np.ndarray,
        weight_values: np.ndarray,
    ) -> np.ndarray:
        """
        Compute expectation values of the observable using the state‑vector
        estimator (no shots).
        """
        param_dict = dict(zip(self.rotation_params, input_values))
        param_dict.update(dict(zip(self.entangle_params, weight_values)))

        return self.estimator.run(
            circuit=self.circuit,
            parameter_binds=[param_dict],
            observables=self.observable,
        )[0][0]


__all__ = ["HybridEstimatorQNN"]
