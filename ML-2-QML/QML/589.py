"""Quantum estimator with 2‑qubit variational circuit and multi‑observable readout."""

from __future__ import annotations

from typing import Optional, Sequence
import numpy as np
from qiskit import QuantumRegister
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import Estimator as StatevectorEstimator

class EstimatorQNNGen236:
    """
    Quantum feed‑forward estimator.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the ansatz.
    hidden_layers : int
        Depth of the entanglement layers.
    input_params : Sequence[Parameter] | None
        Parameters that encode data.
    weight_params : Sequence[Parameter] | None
        Trainable parameters.
    observables : Sequence[SparsePauliOp] | None
        Pauli operators whose expectation values are returned.
    """
    def __init__(self,
                 num_qubits: int = 2,
                 hidden_layers: int = 1,
                 input_params: Optional[list[Parameter]] = None,
                 weight_params: Optional[list[Parameter]] = None,
                 observables: Optional[list[SparsePauliOp]] = None) -> None:
        self.num_qubits = num_qubits
        self.hidden_layers = hidden_layers
        self.input_params = input_params or [Parameter(f"inp{i}") for i in range(num_qubits)]
        self.weight_params = weight_params or [Parameter(f"w{i}") for i in range(num_qubits * hidden_layers)]
        self.observables = observables or [SparsePauliOp.from_list([("Z"*num_qubits, 1)])]
        self.circuit = self._build_circuit()
        self.estimator = StatevectorEstimator()
        self.qnn = QiskitEstimatorQNN(
            circuit=self.circuit,
            observables=self.observables,
            input_params=self.input_params,
            weight_params=self.weight_params,
            estimator=self.estimator
        )

    def _build_circuit(self) -> QuantumCircuit:
        """Construct a layered entangling ansatz."""
        qr = QuantumRegister(self.num_qubits)
        qc = QuantumCircuit(qr)
        # Encode data via Ry on each qubit
        for i, p in enumerate(self.input_params):
            qc.ry(p, i)
        # Entangling layers
        for l in range(self.hidden_layers):
            for i in range(self.num_qubits):
                qc.rx(self.weight_params[l*self.num_qubits + i], i)
            # CNOT chain for entanglement
            for i in range(self.num_qubits - 1):
                qc.cx(i, i+1)
        return qc

    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Evaluate the circuit for a batch of data points.

        Parameters
        ----------
        data : np.ndarray, shape (batch, num_qubits)
            Input data encoded into rotation angles.

        Returns
        -------
        np.ndarray, shape (batch,)
            Expectation values of the specified observables.
        """
        param_dicts = [{p: float(v) for p, v in zip(self.input_params, row)} for row in data]
        results = self.qnn.predict(param_dicts)
        return np.array([res[0].real for res in results])

    def gradient(self, data: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the output w.r.t. weight parameters
        using the parameter‑shift rule.
        """
        param_dicts = [{p: float(v) for p, v in zip(self.input_params, row)} for row in data]
        grads = self.qnn.gradient(param_dicts)
        num_weights = len(self.weight_params)
        grad_arr = np.zeros((data.shape[0], num_weights))
        for i, gdict in enumerate(grads):
            for j, wp in enumerate(self.weight_params):
                grad_arr[i, j] = gdict[wp].real
        return grad_arr

def EstimatorQNN() -> EstimatorQNNGen236:
    """Factory returning a default instance."""
    return EstimatorQNNGen236()

__all__ = ["EstimatorQNNGen236", "EstimatorQNN"]
