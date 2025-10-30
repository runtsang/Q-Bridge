"""Quantum estimator that implements a 1‑qubit parameterised circuit.

The circuit mirrors the simple EstimatorQNN quantum seed but adds
input encoding via an Ry gate and a learnable weight via an Rx
gate.  An expectation value of the Pauli‑Y observable is returned.
The design is deliberately lightweight to allow fast simulation or
execution on real hardware.

Key features:
- Uses Qiskit StatevectorEstimator for fast simulation.
- Supports clipping of the weight parameter to keep the circuit stable.
- Exposes a callable interface that can be used as a drop‑in
  component in a PyTorch model.

Usage
-----
>>> from EstimatorQNN__gen195_qml import EstimatorQNNGenQuantum
>>> q_est = EstimatorQNNGenQuantum()
>>> q_est(0.5)          # returns a torch tensor of shape (1, 1)
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator


def _clip(value: float, bound: float) -> float:
    """Clip a scalar to the range [-bound, bound]."""
    return max(-bound, min(bound, value))


class EstimatorQNNGenQuantum:
    """
    1‑qubit quantum estimator with input encoding and weight.

    Parameters
    ----------
    clip_bound : float, optional
        Upper bound for clipping the weight parameter.  A value of 0
        disables clipping.
    """

    def __init__(self, clip_bound: float = 5.0) -> None:
        self.clip_bound = clip_bound

        # Define circuit parameters
        self.input_param = Parameter("input")
        self.weight_param = Parameter("weight")

        # Build the circuit
        self.circuit = QuantumCircuit(1)
        self.circuit.h(0)
        self.circuit.ry(self.input_param, 0)
        self.circuit.rx(self.weight_param, 0)

        # Observable: Pauli‑Y
        self.observable = SparsePauliOp.from_list([("Y" * self.circuit.num_qubits, 1)])

        # Estimator primitive
        self.estimator = StatevectorEstimator()

        # Wrap into Qiskit‑ML EstimatorQNN
        self.estimator_qnn = EstimatorQNN(
            circuit=self.circuit,
            observables=self.observable,
            input_params=[self.input_param],
            weight_params=[self.weight_param],
            estimator=self.estimator,
        )

    def __call__(self, weight: float, *, input_value: float = 0.0) -> Tensor:
        """
        Evaluate the quantum circuit.

        Parameters
        ----------
        weight : float
            Weight parameter fed to the Rx gate.
        input_value : float, optional
            Value for the Ry input gate.  Defaults to 0.0.

        Returns
        -------
        torch.Tensor
            Expectation value of the Pauli‑Y observable, wrapped in a
            1‑dimensional torch tensor.
        """
        # Clip the weight
        clipped_weight = _clip(weight, self.clip_bound)

        # Execute the circuit
        result = self.estimator_qnn(
            input_values=[input_value],
            weight_values=[clipped_weight],
        )

        # Convert to torch tensor
        expectation = torch.tensor(result[0][0].real, dtype=torch.float32).view(1, 1)
        return expectation


__all__ = ["EstimatorQNNGenQuantum"]
