"""Quantum module for hybrid autoencoder.

Provides:
* QuantumLatentLayer – parameterised circuit that maps a classical latent
  vector into a probability distribution via a StatevectorSampler.
* FastEstimator – wrapper around a qiskit QuantumCircuit for evaluating
  expectation values of observables with optional shot noise.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence

import torch
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator

class QuantumLatentLayer:
    """
    Wraps a Qiskit parameterised circuit that implements a simple 2‑qubit
    ansatz. The circuit accepts two input parameters (the latent vector)
    and a trainable weight vector.  The StatevectorSampler is used to
    exactly compute the resulting probability distribution.
    """
    def __init__(self, num_qubits: int = 2, reps: int = 2) -> None:
        self.num_qubits = num_qubits
        self.reps = reps
        self.circuit = self._build_circuit()
        self.sampler = StatevectorSampler()

    def _build_circuit(self) -> QuantumCircuit:
        inputs = ParameterVector("x", self.num_qubits)
        weights = ParameterVector("w", self.num_qubits * self.reps)
        qc = QuantumCircuit(self.num_qubits)
        # Input layer
        for i in range(self.num_qubits):
            qc.ry(inputs[i], i)
        # Variational part
        idx = 0
        for _ in range(self.reps):
            for i in range(self.num_qubits):
                qc.ry(weights[idx], i)
                idx += 1
            for i in range(self.num_qubits - 1):
                qc.cx(i, i + 1)
        return qc

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs : torch.Tensor
            Shape (batch, num_qubits) – the classical latent vector.
        Returns
        -------
        torch.Tensor
            Shape (batch, 2**num_qubits) – probability distribution from
            the quantum circuit.
        """
        batch, _ = inputs.shape
        probs = []
        for i in range(batch):
            # Bind parameters
            circ = self.circuit.assign_parameters(
                {self.circuit.parameters[0]: inputs[i, 0].item(),
                 self.circuit.parameters[1]: inputs[i, 1].item()})
            state = self.sampler.run(circ).result().quasi_dists[0]
            probs.append(state)
        return torch.tensor(np.array(probs), dtype=torch.float32)

    @property
    def output_dim(self) -> int:
        return 2 ** self.num_qubits

class FastEstimator:
    """
    Very small estimator that evaluates expectation values of
    Pauli‑Z‑like observables for a given parameterized circuit.
    """
    def __init__(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

__all__ = ["QuantumLatentLayer", "FastEstimator"]
