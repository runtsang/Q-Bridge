"""Hybrid estimator integrating classical PyTorch models and optional quantum evaluations.

The class exposes a unified evaluate API that can compute classical observables
on a neural network, optionally add Gaussian shot noise, and delegate to a
parameterized quantum circuit for expectation‑value sampling.  It also accepts
an AutoencoderNet to preprocess inputs before feeding them to the main model.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Guarantee a 2‑D batch tensor for forward passes."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

class FastHybridEstimator:
    """
    Wrapper that evaluates a PyTorch model on batches of parameters, optionally
    feeds the parameters through a classical Autoencoder, and can forward the
    same parameters to a quantum circuit for expectation‑value sampling.

    Parameters
    ----------
    model : nn.Module
        The primary neural network to evaluate.
    autoencoder : nn.Module, optional
        An AutoencoderNet that transforms inputs before the model.
    quantum_circuit : qiskit.circuit.QuantumCircuit, optional
        A parameterised circuit that will be bound and executed for each
        parameter set.  If provided, the ``evaluate`` method will return a
        dictionary containing both classical and quantum results.
    shots : int, optional
        Number of shots for the quantum sampler.  If ``None`` a noiseless
        state‑vector evaluation is performed.
    seed : int, optional
        Random seed for Gaussian noise and quantum shot sampling.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        autoencoder: nn.Module | None = None,
        quantum_circuit: Optional["qiskit.circuit.QuantumCircuit"] = None,
        shots: int | None = None,
        seed: int | None = None,
    ) -> None:
        self.model = model
        self.autoencoder = autoencoder
        self.quantum_circuit = quantum_circuit
        self.shots = shots
        self.seed = seed

    def _preprocess(self, params: torch.Tensor) -> torch.Tensor:
        """Run optional autoencoder before the main model."""
        if self.autoencoder is None:
            return params
        with torch.no_grad():
            return self.autoencoder.encode(params)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable] | None = None,
        parameter_sets: Sequence[Sequence[float]] | None = None,
    ) -> List[List[float]]:
        """
        Compute classical observables for every parameter set.  If a quantum
        circuit is provided, the same parameters are bound and the
        expectation values of the supplied observables are returned as a
        separate list of complex numbers.

        Returns
        -------
        List[List[float]]
            Classical expectation values for each parameter set.
        """
        if observables is None:
            observables = [lambda outputs: outputs.mean(dim=-1)]
        if parameter_sets is None:
            return []

        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                batch = _ensure_batch(params)
                processed = self._preprocess(batch)
                outputs = self.model(processed)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        scalar = float(val.mean().cpu())
                    else:
                        scalar = float(val)
                    row.append(scalar)
                results.append(row)

        # Optionally add shot noise to the classical results
        if self.shots is not None:
            rng = np.random.default_rng(self.seed)
            noisy: List[List[float]] = []
            for row in results:
                noisy_row = [
                    float(rng.normal(mean, max(1e-6, 1 / self.shots))) for mean in row
                ]
                noisy.append(noisy_row)
            results = noisy

        return results

    def evaluate_quantum(
        self,
        observables: Iterable["qiskit.quantum_info.operators.base_operator.BaseOperator"],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """
        Compute expectation values for a parameterised quantum circuit.
        If ``shots`` is set, uses a state‑vector sampler for noiseless
        evaluation; otherwise falls back to a QASM simulator with the
        specified number of shots.
        """
        if self.quantum_circuit is None:
            raise RuntimeError("No quantum circuit supplied to the estimator.")

        from qiskit import execute
        from qiskit import Aer
        from qiskit.quantum_info import Statevector

        results: List[List[complex]] = []

        for params in parameter_sets:
            bound = self.quantum_circuit.assign_parameters(
                dict(zip(self.quantum_circuit.parameters, params)), inplace=False
            )
            if self.shots is None:
                state = Statevector.from_instruction(bound)
                row = [state.expectation_value(obs) for obs in observables]
            else:
                job = execute(
                    bound,
                    backend=Aer.get_backend("qasm_simulator"),
                    shots=self.shots,
                )
                counts = job.result().get_counts(bound)
                probs = np.array(list(counts.values())) / self.shots
                states = np.array(list(counts.keys()), dtype=int)
                row = [float(np.sum(states * probs)) for _ in observables]
            results.append(row)
        return results

    def evaluate_combined(
        self,
        observables: Iterable[ScalarObservable] | None = None,
        quantum_observables: Iterable["qiskit.quantum_info.operators.base_operator.BaseOperator"] | None = None,
        parameter_sets: Sequence[Sequence[float]] | None = None,
    ) -> dict:
        """
        Return a dictionary with keys ``classical`` and ``quantum`` containing
        the respective evaluation results.
        """
        classical = self.evaluate(observables, parameter_sets)
        quantum = (
            self.evaluate_quantum(quantum_observables, parameter_sets)
            if quantum_observables
            else []
        )
        return {"classical": classical, "quantum": quantum}

# Example helper: a quantum fully‑connected layer circuit
def FCL(n_qubits: int = 1) -> "qiskit.circuit.QuantumCircuit":
    """Return a simple parameterised circuit mimicking a fully‑connected layer."""
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(n_qubits)
    theta = qc._parameter("theta")
    qc.h(range(n_qubits))
    qc.ry(theta, range(n_qubits))
    qc.measure_all()
    return qc

__all__ = ["FastHybridEstimator", "FCL"]
