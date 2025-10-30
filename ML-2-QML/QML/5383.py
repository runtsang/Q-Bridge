"""Hybrid quantum-classical binary classifier with a graph-based quantum head.

This module integrates a parameterised two-qubit circuit (Qiskit), a classical
graph neural network (mirrored from GraphQNN), and FastEstimator for shot-noise
evaluation.  The quantum head replaces the classical linear layer of the
classical hybrid classifier.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from typing import Callable, Iterable, Sequence, Tuple

import qiskit
from qiskit import assemble, transpile
from qiskit.providers.aer import AerSimulator

# ---------- Quantum utilities ----------
class QuantumCircuitWrapper:
    """Two-qubit parameterised circuit executed on a Qiskit Aer simulator."""
    def __init__(self, n_qubits: int = 2, shots: int = 8192):
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = AerSimulator()
        self.theta = qiskit.circuit.Parameter("theta")
        self._build()

    def _build(self) -> None:
        self.circuit.h(range(self.n_qubits))
        self.circuit.barrier()
        self.circuit.ry(self.theta, range(self.n_qubits))
        self.circuit.measure_all()

    def run(self, theta: float) -> float:
        """Return the expectation value of Z on the first qubit."""
        compiled = transpile(self.circuit, self.backend)
        bound = compiled.bind_parameters({self.theta: theta})
        job = self.backend.run(assemble(bound, shots=self.shots))
        result = job.result()
        counts = result.get_counts()
        exp = 0.0
        for bitstring, cnt in counts.items():
            z = 1 if bitstring[-1] == "0" else -1
            exp += z * cnt
        return exp / self.shots

# ---------- FastEstimator utilities ----------
ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

class FastBaseEstimator:
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
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
        return results

class FastEstimator(FastBaseEstimator):
    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

# ---------- Graph utilities for embedding ----------
def _random_linear(in_features: int, out_features: int) -> torch.Tensor:
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_graph_network(qnn_arch: Sequence[int], samples: int) -> Tuple[Sequence[int], List[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
    weights: List[torch.Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1]
    training_data = []
    for _ in range(samples):
        features = torch.randn(target_weight.size(1))
        target = target_weight @ features
        training_data.append((features, target))
    return qnn_arch, weights, training_data, target_weight

# ---------- Hybrid classifier ----------
class HybridBinaryClassifier(nn.Module):
    """
    Hybrid quantum-classical binary classifier.
    The input is passed through a lightweight graph neural network
    and then fed into a single-parameter two-qubit circuit that
    produces the class probability via an expectation measurement.
    """
    def __init__(self, num_features: int, qnn_arch: Sequence[int]) -> None:
        super().__init__()
        self.graph_arch = qnn_arch
        self.graph_weights = nn.ParameterList(
            [nn.Parameter(_random_linear(in_f, out_f)) for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:])]
        )
        self.qc = QuantumCircuitWrapper(n_qubits=2, shots=8192)
        self.shift = np.pi / 2  # parameter shift for central difference

    def _graph_forward(self, x: torch.Tensor) -> torch.Tensor:
        activations = x
        for weight in self.graph_weights:
            activations = torch.tanh(weight @ activations.t()).t()
        return activations

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (batch, num_features)
        graph_out = self._graph_forward(inputs)
        theta = graph_out.mean(dim=-1)
        probs = []
        for th in theta:
            exp_z = self.qc.run(float(th.item()))
            p = 0.5 * (1 + exp_z)  # map expectation to probability
            probs.append(p)
        probs = torch.tensor(probs, dtype=torch.float32, device=inputs.device)
        return torch.cat((probs.unsqueeze(-1), (1 - probs).unsqueeze(-1)), dim=-1)

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(inputs)

    def evaluate(self, inputs: torch.Tensor, *, shots: int | None = None, seed: int | None = None) -> List[float]:
        estimator_cls = FastEstimator if shots else FastBaseEstimator
        estimator = estimator_cls(self)
        observables = [lambda out: out[:,0]]
        parameter_sets = [input.tolist() for input in inputs]
        results = estimator.evaluate(observables, parameter_sets, shots=shots, seed=seed)
        return [row[0] for row in results]

__all__ = [
    "HybridBinaryClassifier",
    "FastBaseEstimator",
    "FastEstimator",
]
