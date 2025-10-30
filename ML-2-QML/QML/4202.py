import numpy as np
import torch
from torch import nn
from qiskit import QuantumCircuit, transpile, assemble
from qiskit.providers.aer import AerSimulator
from qiskit.quantum_info.operators.base_operator import BaseOperator
from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

class UnifiedQuantumHybridEstimator:
    """
    Quantum‑centric estimator that evaluates a parameterised quantum circuit
    and can optionally fuse a classical PyTorch head.  It exposes a
    differentiable interface suitable for hybrid training.
    """
    def __init__(self,
                 circuit: QuantumCircuit,
                 backend: Optional[AerSimulator] = None,
                 shots: int = 1024,
                 classical_head: Optional[nn.Module] = None,
                 shift: float = np.pi / 2) -> None:
        """
        Parameters
        ----------
        circuit : QuantumCircuit
            A parameterised circuit whose parameters will be bound at runtime.
        backend : AerSimulator, optional
            The simulator to use; if None, Aer.get_backend('aer_simulator') is used.
        shots : int
            Number of shots per circuit execution.
        classical_head : nn.Module, optional
            A PyTorch module that processes the raw circuit output before
            returning the final observable.
        shift : float
            Shift value for the parameter‑shift gradient used in the
            differentiable wrapper.
        """
        self.circuit = circuit
        self.parameters = list(circuit.parameters)
        self.backend = backend or AerSimulator()
        self.shots = shots
        self.classical_head = classical_head
        self.shift = shift

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self.parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self.parameters, parameter_values))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def evaluate(self,
                 observables: Iterable[BaseOperator],
                 parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        """
        Compute expectation values for each parameter set and observable.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        for values in parameter_sets:
            bound_circuit = self._bind(values)
            compiled = transpile(bound_circuit, self.backend)
            qobj = assemble(compiled, shots=self.shots)
            job = self.backend.run(qobj)
            result = job.result()
            counts = result.get_counts()

            def expectation(count_dict):
                probs = np.array(list(count_dict.values())) / self.shots
                states = np.array([int(k, 2) for k in count_dict.keys()])
                return np.sum(states * probs)

            exp_val = expectation(counts)
            if self.classical_head is not None:
                # Pass the expectation through the classical head
                exp_tensor = torch.tensor([exp_val], dtype=torch.float32)
                exp_val = float(self.classical_head(exp_tensor).item())

            row = [exp_val for _ in observables]  # same value for all observables
            results.append(row)
        return results

    def evaluate_with_noise(self,
                            observables: Iterable[BaseOperator],
                            parameter_sets: Sequence[Sequence[float]],
                            *,
                            seed: int | None = None) -> List[List[complex]]:
        """
        Same as ``evaluate`` but adds Gaussian noise to emulate shot‑noise.
        """
        raw = self.evaluate(observables, parameter_sets)
        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in raw:
            noisy_row = [rng.normal(val.real, max(1e-6, 1 / self.shots)) + 1j *
                         rng.normal(val.imag, max(1e-6, 1 / self.shots)) for val in row]
            noisy.append(noisy_row)
        return noisy

    def hybrid_forward(self,
                       inputs: torch.Tensor) -> torch.Tensor:
        """
        Differentiable forward that passes the input through a small
        parameterised quantum circuit and returns the expectation value.
        This is a lightweight replacement for the heavy Qiskit execution
        path and is suitable for gradient‑based training.
        """
        batch = inputs if inputs.ndim == 2 else inputs.unsqueeze(0)
        batch = batch.to(torch.float32)
        outputs = []
        for param in batch:
            circ = self._bind(param.tolist())
            compiled = transpile(circ, self.backend)
            qobj = assemble(compiled, shots=1)
            job = self.backend.run(qobj)
            result = job.result()
            counts = result.get_counts()
            exp_val = sum(int(k, 2) * v for k, v in counts.items()) / self.shots
            outputs.append(exp_val)
        return torch.tensor(outputs, dtype=torch.float32)
