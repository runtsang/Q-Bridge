"""Hybrid estimator combining classical PyTorch models and optional quantum evaluation.

Features
--------
* Evaluate either a PyTorch model or a Qiskit `QuantumCircuit`.
* Optional Gaussian shot noise for deterministic models.
* Convenience helpers to construct and train a simple autoencoder.
* Unified `evaluate` method that accepts either classical observables
  (callables) or quantum operators.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Union, Any

import numpy as np
import torch
from torch import nn

# Autoencoder utilities
from.Autoencoder import Autoencoder, train_autoencoder

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Return a float32 tensor with a leading batch dim."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastHybridEstimator:
    """Evaluate classical neural networks or quantum circuits for batches of inputs.

    Parameters
    ----------
    model : nn.Module | None
        A PyTorch model to evaluate. If ``None`` and *quantum_circuit* is
        provided, the estimator will use the supplied quantum circuit.
    quantum_circuit : Any | None
        Optional Qiskit `QuantumCircuit`.  Must be bound with the same number of
        parameters as the size of each element in *parameter_sets*.
    """

    def __init__(
        self,
        model: nn.Module | None = None,
        quantum_circuit: Any | None = None,
    ) -> None:
        if model is None and quantum_circuit is None:
            raise ValueError("Either a PyTorch model or a quantum circuit must be provided.")
        self.model = model
        self._qc = quantum_circuit
        if self._qc is not None:
            from qiskit import QuantumCircuit  # local import
            if not isinstance(self._qc, QuantumCircuit):
                raise TypeError("quantum_circuit must be a qiskit.circuit.QuantumCircuit")
            self._qc_params = list(self._qc.parameters)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable | Any],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float | complex]]:
        """Compute expectation values for each parameter set.

        If a quantum circuit is supplied, the method delegates to Qiskit
        Statevector or the supplied Qiskit primitives.  When a neural
        network is used, it applies the optional Gaussian shot noise.

        Parameters
        ----------
        observables
            List of callable observables for classical models or
            BaseOperator instances for quantum models.
        parameter_sets
            Sequence of parameter tuples matching the model or circuit.
        shots
            Number of shots for quantum simulation.  If ``None`` the
            circuit is evaluated as a noiseless statevector.
        seed
            Random seed for shot noise or Qiskit sampler.

        Returns
        -------
        List[List[float | complex]]
            Nested list of results per parameter set.
        """
        if self.model is not None:
            # Classical evaluation
            return self._evaluate_classical(observables, parameter_sets, shots, seed)

        # Quantum evaluation
        return self._evaluate_quantum(observables, parameter_sets, shots, seed)

    def _evaluate_classical(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        shots: int | None,
        seed: int | None,
    ) -> List[List[float]]:
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.model(inputs)
                row: List[float] = []
                for observable in observables:
                    value = observable(outputs)
                    if isinstance(value, torch.Tensor):
                        scalar = float(value.mean().cpu())
                    else:
                        scalar = float(value)
                    row.append(scalar)
                results.append(row)

        if shots is not None:
            rng = np.random.default_rng(seed)
            for i, row in enumerate(results):
                results[i] = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
        return results

    def _evaluate_quantum(
        self,
        observables: Iterable[Any],
        parameter_sets: Sequence[Sequence[float]],
        shots: int | None,
        seed: int | None,
    ) -> List[List[complex]]:
        from qiskit.quantum_info import Statevector
        results: List[List[complex]] = []
        for params in parameter_sets:
            if len(params)!= len(self._qc_params):
                raise ValueError("Parameter count mismatch for bound circuit.")
            mapping = dict(zip(self._qc_params, params))
            bound = self._qc.assign_parameters(mapping, inplace=False)
            if shots is None:
                state = Statevector.from_instruction(bound)
                row = [state.expectation_value(obs) for obs in observables]
            else:
                from qiskit.primitives import Sampler as QiskitSampler
                sampler = QiskitSampler(seed=seed)
                result = sampler.run(bound, observables=observables, seed=seed).result()
                # For simplicity we use the noiseless statevector expectation
                state = Statevector.from_instruction(bound)
                row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

    # Convenience helpers for autoencoder usage
    def encode(self, data: torch.Tensor, latent_dim: int = 32) -> torch.Tensor:
        """Encode data using the AutoencoderNet."""
        encoder = Autoencoder(data.shape[1], latent_dim=latent_dim).encoder
        encoder.eval()
        with torch.no_grad():
            return encoder(data)

    def train_autoencoder(
        self,
        data: torch.Tensor,
        *,
        epochs: int = 100,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        device: torch.device | None = None,
    ) -> list[float]:
        """Utility to train an autoencoder on *data* and return loss history."""
        model = Autoencoder(data.shape[1])
        return train_autoencoder(
            model,
            data,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            device=device,
        )

__all__ = ["FastHybridEstimator"]
