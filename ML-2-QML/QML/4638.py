"""Quantum estimator used by the hybrid regressor.

The class implements a parameterized quantum circuit that acts as a
regression head.  It uses a single layer of Ry rotations followed by a
chain of CZ entangling gates, and evaluates the expectation value of a
collective Pauli‑Z observable.  The circuit is built with Qiskit and
executed via the Estimator primitive, which makes it compatible with
any qasm or statevector simulator.

The interface mirrors the original EstimatorQNN example, allowing the
classical network to treat the quantum circuit as a parameterized
layer.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator


class HybridQuantumEstimator:
    """
    Parameterized quantum circuit for regression.

    Parameters
    ----------
    n_qubits : int, default=1
        Number of qubits in the circuit.
    shots : int, default=1024
        Number of shots for the simulator.
    """

    def __init__(self, n_qubits: int = 1, shots: int = 1024) -> None:
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = qiskit.Aer.get_backend("qasm_simulator")
        self.params = ParameterVector("theta", length=n_qubits)
        self.circuit = self._build_circuit()
        self.estimator = Estimator()
        self.observable = SparsePauliOp.from_list([("Z" * n_qubits, 1.0)])

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        qc.h(range(self.n_qubits))
        for i, p in enumerate(self.params):
            qc.ry(p, i)
        # Entangling layer
        for i in range(self.n_qubits - 1):
            qc.cz(i, i + 1)
        return qc

    def run(self, input_params, weight_params):
        """
        Evaluate the quantum circuit.

        Parameters
        ----------
        input_params : list, optional
            List of input parameters (currently unused for this circuit).
        weight_params : list of float
            Rotation angles for the Ry gates.

        Returns
        -------
        float
            Expectation value of the observable.
        """
        bind_dict = {self.params[i]: weight_params[i] for i in range(self.n_qubits)}
        bound_qc = self.circuit.bind_parameters(bind_dict)

        # Run with Estimator
        result = self.estimator.run(
            circuits=[bound_qc],
            observables=[self.observable],
            parameter_values=[weight_params],
        )
        # result is a list of Result objects
        return float(result[0].values[0])


__all__ = ["HybridQuantumEstimator"]
