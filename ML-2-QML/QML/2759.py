import numpy as np
import torch
from torch import nn
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from collections.abc import Iterable, Sequence
from typing import List, Optional

class HybridEstimator:
    """
    Quantum estimator that evaluates expectation values of observables for a parametrized circuit.
    Supports shot‑noise simulation and optional classical neural‑network weight generation via EstimatorQNN.
    """
    def __init__(self, circuit: QuantumCircuit, observables: Iterable[BaseOperator], shots: Optional[int] = None) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)
        self._observables = list(observables)
        self._shots = shots
        self._weight_net: Optional[nn.Module] = None

    def set_weight_net(self, net: nn.Module) -> None:
        """Attach a classical neural network that produces weight parameters from input parameters."""
        self._weight_net = net

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(self, parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        """
        Compute expectation values for each parameter set and observable.
        If a weight network is attached, its outputs are used as weight parameters.
        """
        results: List[List[complex]] = []
        for values in parameter_sets:
            if self._weight_net is not None:
                # Generate weight params using the neural network
                input_tensor = torch.as_tensor(values, dtype=torch.float32).unsqueeze(0)
                weight_params = self._weight_net(input_tensor).detach().numpy().flatten().tolist()
                # Split parameters: first len(weight_params) are weights, rest are inputs
                mapping = dict(zip(self._parameters[:len(weight_params)], weight_params))
                mapping.update(dict(zip(self._parameters[len(weight_params):], values)))
                bound_circ = self._circuit.assign_parameters(mapping, inplace=False)
            else:
                bound_circ = self._bind(values)
            state = Statevector.from_instruction(bound_circ)
            row = [state.expectation_value(obs) for obs in self._observables]
            if self._shots is not None:
                noisy_row = []
                for exp in row:
                    mean = exp.real
                    std = 1 / np.sqrt(self._shots)
                    noisy_val = mean + np.random.normal(0, std)
                    noisy_row.append(complex(noisy_val, 0))
                row = noisy_row
            results.append(row)
        return results

__all__ = ["HybridEstimator"]
