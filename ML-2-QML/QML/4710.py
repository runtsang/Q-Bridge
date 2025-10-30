"""Quantum estimator that fuses a self‑attention style entanglement block
with a variational ansatz and a state‑vector estimator.

The circuit layout (4 qubits) is:
    - Encode the two classical inputs on qubits 0 and 1 via Ry gates.
    - A parameterized self‑attention block consisting of Rx/Ry/Rz rotations
      followed by controlled‑Rx entanglement between adjacent qubits.
    - A variational layer of Rx/Ry/Rz rotations and CNOTs to increase expressivity.
    - Observable: sum of Pauli‑Z on all qubits.
The class exposes a ``forward`` method compatible with the classical
EstimatorQNNGen157, enabling a seamless switch between classical and quantum backends.
"""

from __future__ import annotations

import math
from typing import List

import numpy as np
import qiskit
from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator


class EstimatorQNNGen157:
    """Hybrid quantum estimator combining self‑attention and variational layers.

    The object behaves like a neural network: ``forward(x)`` accepts a
    NumPy array of shape (batch, 2) and returns predictions.  Internally
    it uses a :class:`~qiskit_machine_learning.neural_networks.EstimatorQNN`
    with a custom circuit.
    """
    def __init__(self) -> None:
        self.estimator = self._build_estimator()

    @staticmethod
    def _build_estimator() -> EstimatorQNN:
        # Input parameters (classical features)
        input_params: List[Parameter] = [Parameter("x0"), Parameter("x1")]

        # Weight parameters for self‑attention rotations, entanglement and variational block
        weight_params: List[Parameter] = []

        # Self‑attention rotations (Rx, Ry, Rz for each qubit)
        for i in range(4):
            weight_params.extend(
                [Parameter(f"rx_{i}"), Parameter(f"ry_{i}"), Parameter(f"rz_{i}")]
            )

        # Entanglement via controlled‑Rx between adjacent qubits
        for i in range(3):
            weight_params.append(Parameter(f"crx_{i}"))

        # Variational rotations
        for i in range(4):
            weight_params.extend(
                [Parameter(f"vx_{i}"), Parameter(f"vy_{i}"), Parameter(f"vz_{i}")]
            )

        # Build the circuit
        qc = QuantumCircuit(4)

        # Encode inputs on qubits 0 and 1
        qc.ry(input_params[0], 0)
        qc.ry(input_params[1], 1)

        # Self‑attention rotations
        idx = 0
        for qubit in range(4):
            qc.rx(weight_params[idx], qubit)
            qc.ry(weight_params[idx + 1], qubit)
            qc.rz(weight_params[idx + 2], qubit)
            idx += 3

        # Entanglement
        for i in range(3):
            qc.crx(weight_params[idx], i, i + 1)
            idx += 1

        # Variational rotations
        for qubit in range(4):
            qc.rx(weight_params[idx], qubit)
            qc.ry(weight_params[idx + 1], qubit)
            qc.rz(weight_params[idx + 2], qubit)
            idx += 3

        # Simple entangling pattern: CNOT chain
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 3)

        # Observable: sum of Pauli‑Z on all qubits
        observable = SparsePauliOp.from_list([("Z" * qc.num_qubits, 1)])

        # Primitive estimator (state‑vector)
        estimator = StatevectorEstimator()

        # Construct the EstimatorQNN wrapper
        estimator_qnn = EstimatorQNN(
            circuit=qc,
            observables=observable,
            input_params=input_params,
            weight_params=weight_params,
            estimator=estimator,
        )
        return estimator_qnn

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Predict on a batch of 2‑dimensional inputs.

        Parameters
        ----------
        x : np.ndarray, shape (batch, 2)
            Classical feature matrix.

        Returns
        -------
        np.ndarray, shape (batch, 1)
            Predicted values from the quantum circuit.
        """
        if x.ndim == 1:
            x = x[None]
        return self.estimator.predict(x).reshape(-1, 1)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


__all__ = ["EstimatorQNNGen157"]
