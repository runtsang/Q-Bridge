"""Hybrid estimator that can evaluate both PyTorch models and Qiskit circuits.

The estimator accepts a model that is either a torch.nn.Module or a
qiskit.circuit.QuantumCircuit.  It exposes a single evaluate method that
takes a list of observables and a batch of parameter sets.  When the model
is quantum, the evaluation uses a state‑vector simulator and supports a
shots argument to perform stochastic sampling.  For classical models the
method optionally adds Gaussian shot noise to emulate measurement statistics.

The class also provides helper constructors for the fraud‑detection model
and a simple self‑attention block, mirroring the functionality of the
reference seeds while keeping the public API compact.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Callable, List, Union

import numpy as np
import torch
from torch import nn
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]
QuantumObservable = Callable[[Statevector], complex | float]
Observable = Union[ScalarObservable, QuantumObservable]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Return a 2‑D tensor of shape (batch, 1) for a single value or
    shape (batch, 2) for 2‑D inputs.  The function is generic enough to
    wrap both numpy arrays and Python lists.
    """
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


class HybridEstimator:
    """
    Hybrid estimator that can evaluate both PyTorch models and Qiskit circuits.

    Parameters
    ----------
    model : nn.Module | QuantumCircuit
        The model to evaluate.  Either a PyTorch neural network or a
        parameterised Qiskit circuit.
    """

    def __init__(self, model: Union[nn.Module, QuantumCircuit]) -> None:
        if isinstance(model, nn.Module):
            self._model = model
            self._is_quantum = False
        elif isinstance(model, QuantumCircuit):
            self._model = model
            self._is_quantum = True
        else:
            raise TypeError(
                "model must be torch.nn.Module or qiskit.circuit.QuantumCircuit"
            )
        self._backend = None

    @staticmethod
    def _default_observable(outputs: torch.Tensor) -> torch.Tensor:
        """Fallback observable returning the mean of the output tensor."""
        return outputs.mean(dim=-1)

    def _bind_circuit(self, params: Sequence[float]) -> QuantumCircuit:
        """Bind a list of parameter values to the quantum circuit."""
        if len(params)!= len(self._model.parameters()):
            raise ValueError("Parameter count mismatch for circuit binding.")
        mapping = dict(zip(self._model.parameters(), params))
        return self._model.assign_parameters(mapping, inplace=False)

    def _shots_to_expectation(
        self, counts: dict[str, int], observable: BaseOperator, shots: int
    ) -> float:
        """
        Convert raw counts to expectation value for a Pauli‑Z like observable.
        A generic implementation is provided; specialised observables can be
        handled by subclassing or overriding this method.
        """
        exp = 0.0
        for bitstring, freq in counts.items():
            parity = (-1) ** sum(int(b) for b in bitstring)
            exp += parity * freq
        return exp / shots

    def evaluate(
        self,
        observables: Iterable[Observable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """
        Evaluate the wrapped model for each parameter set and observable.

        Parameters
        ----------
        observables : iterable
            For a classical model: callables that map a model output tensor to a scalar.
            For a quantum model: :class:`qiskit.quantum_info.operators.base_operator.BaseOperator`
            instances whose expectation value can be computed from a :class:`Statevector`.
        parameter_sets : sequence of sequences
            Batch of parameter values to bind to the model or circuit.
        shots : int, optional
            If supplied and the model is quantum, perform a shot‑based simulation
            using Qiskit Aer.  For classical models the same argument triggers
            Gaussian noise sampling.
        seed : int, optional
            Random seed for noise generation (both classical and quantum).
        """
        observables = list(observables) or [self._default_observable]
        results: List[List[float]] = []

        if self._is_quantum:
            for params in parameter_sets:
                bound_circuit = self._bind_circuit(params)
                if shots is None:
                    state = Statevector.from_instruction(bound_circuit)
                    row = [float(state.expectation_value(obs)) for obs in observables]
                else:
                    job = execute(bound_circuit, backend=self._backend or "qasm_simulator", shots=shots)
                    counts = job.result().get_counts(bound_circuit)
                    row = [self._shots_to_expectation(counts, obs, shots) for obs in observables]
                results.append(row)
        else:
            self._model.eval()
            with torch.no_grad():
                for params in parameter_sets:
                    inputs = _ensure_batch(params)
                    outputs = self._model(inputs)
                    row: List[float] = []
                    for obs in observables:
                        val = obs(outputs)
                        if isinstance(val, torch.Tensor):
                            scalar = float(val.mean().cpu())
                        else:
                            scalar = float(val)
                        row.append(scalar)
                    results.append(row)

        if shots is not None and not self._is_quantum:
            rng = np.random.default_rng(seed)
            noisy: List[List[float]] = []
            for row in results:
                noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
                noisy.append(noisy_row)
            return noisy

        return results

    @staticmethod
    def build_fraud_detection_program(
        input_params: "HybridEstimator.FraudLayerParameters",
        layers: Iterable["HybridEstimator.FraudLayerParameters"],
    ) -> nn.Sequential:
        """
        Construct a fraud‑detection neural network mirroring the photonic
        architecture.  The first layer is unclipped while subsequent layers
        are clipped to keep the weights within a reasonable range.
        """
        def _layer_from_params(params: "HybridEstimator.FraudLayerParameters", clip: bool) -> nn.Module:
            weight = torch.tensor(
                [[params.bs_theta, params.bs_phi],
                 [params.squeeze_r[0], params.squeeze_r[1]]],
                dtype=torch.float32,
            )
            bias = torch.tensor(params.phases, dtype=torch.float32)
            if clip:
                weight = weight.clamp(-5.0, 5.0)
                bias = bias.clamp(-5.0, 5.0)
            linear = nn.Linear(2, 2)
            with torch.no_grad():
                linear.weight.copy_(weight)
                linear.bias.copy_(bias)
            activation = nn.Tanh()
            scale = torch.tensor(params.displacement_r, dtype=torch.float32)
            shift = torch.tensor(params.displacement_phi, dtype=torch.float32)

            class Layer(nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.linear = linear
                    self.activation = activation
                    self.register_buffer("scale", scale)
                    self.register_buffer("shift", shift)

                def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
                    outputs = self.activation(self.linear(inputs))
                    outputs = outputs * self.scale + self.shift
                    return outputs

            return Layer()

        modules: List[nn.Module] = [_layer_from_params(input_params, clip=False)]
        modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
        modules.append(nn.Linear(2, 1))
        return nn.Sequential(*modules)

    @staticmethod
    def build_self_attention(embed_dim: int) -> nn.Module:
        """
        Return a lightweight self‑attention block compatible with the
        original SelfAttention seed.  The implementation uses a simple
        linear projection for query/key/value and a softmax normalisation.
        """
        class ClassicalSelfAttention(nn.Module):
            def __init__(self, embed_dim: int) -> None:
                super().__init__()
                self.embed_dim = embed_dim

            def forward(
                self,
                rotation_params: torch.Tensor,
                entangle_params: torch.Tensor,
                inputs: torch.Tensor,
            ) -> torch.Tensor:
                query = torch.matmul(inputs, rotation_params.reshape(self.embed_dim, -1))
                key = torch.matmul(inputs, entangle_params.reshape(self.embed_dim, -1))
                scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
                return scores @ inputs

        return ClassicalSelfAttention(embed_dim)


__all__ = ["HybridEstimator", "FraudLayerParameters"]
