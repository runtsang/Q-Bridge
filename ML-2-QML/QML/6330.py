"""Hybrid classical-quantum convolutional network for binary classification.

This module exposes a quantum circuit interface and a hybrid neural
network composed of convolutional layers followed by a parameterised
quantum expectation layer.  Fast evaluation utilities are provided
to compute expectation values for a batch of parameter sets with
optional shot noise, mirroring the FastBaseEstimator from the
original repository.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import qiskit
import qiskit.Aer
from qiskit import assemble, transpile
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from collections.abc import Iterable, Sequence
from typing import Optional

class QuantumCircuitWrapper:
    """Wrapper around a parametrised circuit executed on Aer.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    backend : qiskit.providers.baseprovider.Provider
        Backend to run the circuit on.
    shots : int
        Number of shots for simulation.
    """

    def __init__(self, n_qubits: int, backend, shots: int) -> None:
        self._circuit = QuantumCircuit(n_qubits)
        all_qubits = list(range(n_qubits))
        self.theta = qiskit.circuit.Parameter("theta")

        self._circuit.h(all_qubits)
        self._circuit.barrier()
        self._circuit.ry(self.theta, all_qubits)
        self._circuit.measure_all()

        self.backend = backend
        self.shots = shots

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """Execute the parametrised circuit for the provided angles."""
        compiled = transpile(self._circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()

        def expectation(count_dict):
            counts = np.array(list(count_dict.values()))
            states = np.array(list(count_dict.keys())).astype(float)
            probabilities = counts / self.shots
            return np.sum(states * probabilities)

        if isinstance(result, list):
            return np.array([expectation(item) for item in result])
        return np.array([expectation(result)])

class HybridFunction(torch.autograd.Function):
    """Differentiable interface between PyTorch and the quantum circuit."""

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuitWrapper, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.quantum_circuit = circuit

        expectation_z = ctx.quantum_circuit.run(inputs.tolist())
        result = torch.tensor([expectation_z])
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        input_values = np.array(inputs.tolist())
        shift = np.ones_like(input_values) * ctx.shift

        gradients = []
        for idx, value in enumerate(input_values):
            expectation_right = ctx.quantum_circuit.run([value + shift[idx]])
            expectation_left = ctx.quantum_circuit.run([value - shift[idx]])
            gradients.append(expectation_right - expectation_left)

        gradients = torch.tensor([gradients]).float()
        return gradients * grad_output.float(), None, None

class Hybrid(nn.Module):
    """Hybrid layer that forwards activations through a quantum circuit."""

    def __init__(self, n_qubits: int, backend, shots: int, shift: float) -> None:
        super().__init__()
        self.quantum_circuit = QuantumCircuitWrapper(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        squeezed = torch.squeeze(inputs) if inputs.shape!= torch.Size([1, 1]) else inputs[0]
        return HybridFunction.apply(squeezed, self.quantum_circuit, self.shift)

class HybridBinaryClassifier(nn.Module):
    """Convolutional network followed by a quantum expectation head."""

    def __init__(self,
                 conv_filters: Optional[Sequence[int]] = None,
                 fc_sizes: Optional[Sequence[int]] = None,
                 n_qubits: int = 2,
                 shift: float = np.pi / 2,
                 shots: int = 100) -> None:
        super().__init__()
        conv_filters = conv_filters or [6, 15]
        fc_sizes = fc_sizes or [120, 84]
        self.conv1 = nn.Conv2d(3, conv_filters[0], kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(conv_filters[0], conv_filters[1], kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        dummy_input = torch.zeros(1, 3, 252, 252)
        with torch.no_grad():
            x = self.pool(F.relu(self.conv1(dummy_input)))
            x = self.drop1(x)
            x = self.pool(F.relu(self.conv2(x)))
            x = self.drop1(x)
            x = torch.flatten(x, 1)
        flat_features = x.shape[1]
        self.fc1 = nn.Linear(flat_features, fc_sizes[0])
        self.fc2 = nn.Linear(fc_sizes[0], fc_sizes[1])
        self.fc3 = nn.Linear(fc_sizes[1], 1)

        backend = qiskit.Aer.get_backend("aer_simulator")
        self.hybrid = Hybrid(n_qubits, backend, shots, shift)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(inputs))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.hybrid(x).T
        return torch.cat((x, 1 - x), dim=-1)

class FastEstimator:
    """Fast evaluation of a quantum model with optional Gaussian shot noise.

    Parameters
    ----------
    model : nn.Module
        The hybrid model that contains a quantum head.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """Compute expectation values for every parameter set and observable.

        Parameters
        ----------
        observables : iterable of qiskit operators
            Each operator is evaluated on the circuit state produced by
            the model.
        parameter_sets : sequence of parameter vectors
            Each vector is fed as input to the model.
        shots : int, optional
            If supplied, Gaussian noise with variance 1/shots is added to
            each observable value.
        seed : int, optional
            Random seed for reproducible noise.
        """
        observables = list(observables)
        results: List[List[complex]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = torch.as_tensor(params, dtype=torch.float32).unsqueeze(0)
                _ = self.model(inputs)  # forward to ensure internal state

                bound_circuit = self.model.hybrid.quantum_circuit._circuit.assign_parameters(
                    {self.model.hybrid.quantum_circuit.theta: params[0]},
                    inplace=False,
                )
                state = Statevector.from_instruction(bound_circuit)
                row = [state.expectation_value(obs) for obs in observables]
                results.append(row)

        if shots is None:
            return results

        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in results:
            noisy_row = [rng.normal(row_val.real, max(1e-6, 1 / shots)) + 1j * rng.normal(row_val.imag, max(1e-6, 1 / shots))
                         for row_val in row]
            noisy.append(noisy_row)
        return noisy

__all__ = ["QuantumCircuitWrapper", "HybridFunction", "Hybrid", "HybridBinaryClassifier", "FastEstimator"]
