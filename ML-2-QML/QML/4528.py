from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector
from qiskit.primitives import Sampler
import numpy as np
import torch
from typing import Iterable, Sequence, List, Callable
from qiskit.quantum_info.operators import BaseOperator

class HybridSamplerQNN:
    """
    Quantum side of the hybrid sampler.

    The circuit accepts a vector of parameters that is split into
    *classical* (input) and *quantum* (weight) parts.  The classical
    parameters drive single‑qubit rotations that embed the classical
    embedding produced by the PyTorch side; the quantum parameters
    control additional rotations that act as a variational sampler.

    The class implements ``__call__`` to return a probability vector
    and ``evaluate`` to compute expectation values of arbitrary
    BaseOperator observables.
    """
    def __init__(self, n_classical: int = 2, n_quantum: int = 4) -> None:
        self.n_classical = n_classical
        self.n_quantum = n_quantum
        self.inputs = ParameterVector("input", n_classical)
        self.weights = ParameterVector("weight", n_quantum)
        self.circuit = QuantumCircuit(n_classical)
        # Classical embedding: single‑qubit rotations
        for i in range(n_classical):
            self.circuit.ry(self.inputs[i], i)
        # Entanglement
        self.circuit.cx(0, 1)
        # Variational part: quantum weights
        for i in range(n_quantum):
            self.circuit.ry(self.weights[i], i % n_classical)
        # Final entanglement
        self.circuit.cx(0, 1)
        self.sampler = Sampler()

    def _bind(self, param_values: Sequence[float]) -> QuantumCircuit:
        """Return a circuit with parameters bound to the supplied values."""
        if len(param_values)!= self.n_classical + self.n_quantum:
            raise ValueError("Parameter count mismatch")
        mapping = {p: v for p, v in zip(self.inputs + self.weights, param_values)}
        return self.circuit.assign_parameters(mapping, inplace=False)

    def __call__(self, params: torch.Tensor) -> torch.Tensor:
        """
        Sample the probability distribution for a single or batch of
        parameter vectors.

        Parameters
        ----------
        params : torch.Tensor
            Shape ``(batch, n_classical + n_quantum)`` or ``(n_classical + n_quantum,)``.

        Returns
        -------
        torch.Tensor
            Probability vector(s) of shape ``(batch, 2**n_classical)``.
        """
        if params.ndim == 1:
            params = params.unsqueeze(0)
        probs = []
        for row in params:
            circ = self._bind(row.tolist())
            sv = Statevector.from_instruction(circ)
            probs.append(sv.probabilities())
        return torch.tensor(np.array(probs), dtype=torch.float32)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """
        Compute expectation values for a list of observables and
        parameter sets.

        Parameters
        ----------
        observables : iterable of BaseOperator
            Operators whose expectation values are desired.
        parameter_sets : sequence of parameter sequences
            Each entry contains a full set of parameters for a single circuit.

        Returns
        -------
        List[List[complex]]
            Nested list where the outer dimension is the parameter sets
            and the inner dimension the observables.
        """
        results: List[List[complex]] = []
        for values in parameter_sets:
            circ = self._bind(values)
            sv = Statevector.from_instruction(circ)
            row = [sv.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

__all__ = ["HybridSamplerQNN"]
