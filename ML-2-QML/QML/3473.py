"""Quantum‑enhanced estimator with a variational circuit and a quantum kernel.

The implementation mirrors the original EstimatorQNN but extends it with:

* A two‑qubit ansatz that encodes the input on qubit 0 and the weight on qubit 1.
* A Y‑observable measurement that yields a continuous output in ``[-1, 1]``.
* A helper method to compute a quantum kernel matrix via state‑vector overlap,
  re‑using the same ansatz for both data points.

All components are built with Qiskit and its StatevectorEstimator primitive, making
the code runnable on any backend that supports state‑vector simulation.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import torch
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator

__all__ = ["CombinedEstimatorQNN"]


class CombinedEstimatorQNN:
    """
    Quantum estimator that evaluates a variational circuit and optionally
    computes a quantum kernel matrix.

    Parameters
    ----------
    input_dim : int, default 1
        Dimensionality of the classical input.  The circuit uses one qubit for the
        input and one qubit for the weight parameter.
    """

    def __init__(self, input_dim: int = 1) -> None:
        self.input_dim = input_dim
        # Parameters for encoding
        self.input_params = [Parameter(f"x_{i}") for i in range(input_dim)]
        self.weight_param = Parameter("w")

        # Build the circuit
        self.circuit = QuantumCircuit(input_dim + 1)
        # Encode input on qubit 0
        for i in range(input_dim):
            self.circuit.ry(self.input_params[i], i)
        # Entangle with the weight qubit
        for i in range(input_dim):
            self.circuit.cx(i, input_dim)
        # Apply weight rotation on the ancillary qubit
        self.circuit.rx(self.weight_param, input_dim)
        # Measurement observable (Y on qubit 0)
        self.observable = SparsePauliOp.from_list([("Y" + "I" * input_dim, 1)])

        # State‑vector estimator primitive
        self.estimator = StatevectorEstimator()
        self._compiled = False

    # ----------------------------------------------------------------------
    # Core evaluation
    # ----------------------------------------------------------------------
    def evaluate(self, inputs: Sequence[float], weight: float) -> float:
        """
        Evaluate the circuit for a single data point.

        Parameters
        ----------
        inputs : Sequence[float]
            Classical input of length ``input_dim``.
        weight : float
            Weight parameter for the variational ansatz.

        Returns
        -------
        float
            Expected value of the Y observable, in ``[-1, 1]``.
        """
        if len(inputs)!= self.input_dim:
            raise ValueError("Input dimensionality mismatch.")
        param_bindings = {p: v for p, v in zip(self.input_params, inputs)}
        param_bindings[self.weight_param] = weight
        result = self.estimator.run(
            circuit=self.circuit,
            params=param_bindings,
            observables=[self.observable],
        )
        return float(result[0].values[0].real)

    def batch_predict(self, inputs: torch.Tensor, weight: float) -> torch.Tensor:
        """
        Batch evaluation for a tensor of inputs.

        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of shape ``(batch, input_dim)``.
        weight : float
            Weight parameter shared across the batch.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(batch,)`` containing the expectation values.
        """
        preds = torch.empty(inputs.shape[0], dtype=torch.float32)
        for idx, sample in enumerate(inputs):
            preds[idx] = self.evaluate(sample.tolist(), weight)
        return preds

    # ----------------------------------------------------------------------
    # Quantum kernel helpers
    # ----------------------------------------------------------------------
    def _kernel_circuit(self, data: Sequence[float]) -> QuantumCircuit:
        """
        Internal helper that constructs a circuit encoding a single data point
        using only the input parameters (no weight).  The circuit is identical
        to ``self.circuit`` but with the weight qubit reset to |0>.
        """
        qc = QuantumCircuit(self.input_dim + 1)
        for i in range(self.input_dim):
            qc.ry(data[i], i)
        for i in range(self.input_dim):
            qc.cx(i, self.input_dim)
        qc.reset(self.input_dim)  # ensure ancillary qubit is |0>
        return qc

    def quantum_kernel(self, x: Sequence[float], y: Sequence[float]) -> float:
        """
        Compute the overlap between two encoded states |ψ(x)⟩ and |ψ(y)⟩.
        The kernel value is |⟨ψ(x)|ψ(y)⟩|² ∈ [0, 1].

        Parameters
        ----------
        x, y : Sequence[float]
            Classical data points of length ``input_dim``.

        Returns
        -------
        float
            Quantum kernel value.
        """
        qc_x = self._kernel_circuit(x)
        qc_y = self._kernel_circuit(y)
        # Compute statevectors
        sv_x = self.estimator.run(
            qc_x, observables=[SparsePauliOp.from_list([("I" * (self.input_dim + 1), 1)])]
        )[0].values[0]
        sv_y = self.estimator.run(
            qc_y, observables=[SparsePauliOp.from_list([("I" * (self.input_dim + 1), 1)])]
        )[0].values[0]
        overlap = np.vdot(sv_x, sv_y)
        return float(np.abs(overlap) ** 2)

    def kernel_matrix(self, a: Iterable[Sequence[float]], b: Iterable[Sequence[float]]) -> np.ndarray:
        """
        Evaluate the Gram matrix between two collections of data points.

        Parameters
        ----------
        a, b : Iterable[Sequence[float]]
            Collections of data points of shape ``(N, input_dim)`` and ``(M, input_dim)``.

        Returns
        -------
        np.ndarray
            Kernel matrix of shape ``(len(a), len(b))``.
        """
        a = list(a)
        b = list(b)
        return np.array(
            [[self.quantum_kernel(x, y) for y in b] for x in a]
        )
