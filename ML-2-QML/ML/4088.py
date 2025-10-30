"""Unified hybrid layer module.

Combines classical dense head, quantum expectation layer and batched
estimator with optional Gaussian shot noise.  It can be used as a
drop‑in replacement for the original FCL, FastBaseEstimator or
HybridNet modules.

The class is fully differentiable when the quantum circuit is
implemented as a Qiskit Aer simulator and the HybridFunction is
used.  In the classical variant the quantum part is executed
sequentially using the simulator and the output is wrapped in a
torch.Tensor.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from collections.abc import Sequence
from typing import List

# --------------------------------------------------------------------------- #
#  Classical dense block
# --------------------------------------------------------------------------- #
class _DenseBlock(nn.Module):
    """Thin linear layer with tanh activation, mirroring the FCL seed."""
    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.linear(x))

# --------------------------------------------------------------------------- #
#  Quantum circuit wrapper
# --------------------------------------------------------------------------- #
class _QuantumCircuitWrapper:
    """Parameterised Qiskit circuit that returns an expectation value."""
    def __init__(self, n_qubits: int = 1, backend=None, shots: int = 100):
        if backend is None:
            import qiskit
            backend = qiskit.Aer.get_backend("qasm_simulator")
        self.backend = backend
        self.shots = shots
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        self._theta = qiskit.circuit.Parameter("theta")
        self._circuit.h(range(n_qubits))
        self._circuit.barrier()
        self._circuit.ry(self._theta, range(n_qubits))
        self._circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """Execute the circuit for a 1‑D array of angles and return an
        expectation value as a 1‑D numpy array."""
        import qiskit
        job = qiskit.execute(
            self._circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[{self._theta: theta} for theta in thetas],
        )
        result = job.result().get_counts(self._circuit)
        counts = np.array(list(result.values()))
        states = np.array(list(result.keys())).astype(float)
        probabilities = counts / self.shots
        expectation = np.sum(states * probabilities)
        return np.array([expectation])

# --------------------------------------------------------------------------- #
#  Estimator with optional shot noise
# --------------------------------------------------------------------------- #
class _Estimator:
    """Batched evaluator for a parameterised quantum circuit."""
    def __init__(self, circuit: _QuantumCircuitWrapper, noise_shots: int | None = None, seed: int | None = None):
        self.circuit = circuit
        self.noise_shots = noise_shots
        self.seed = seed
        if noise_shots is not None:
            self.rng = np.random.default_rng(seed)

    def evaluate(self, param_sets: Sequence[Sequence[float]]) -> List[float]:
        """Return a list of expectation values for each parameter set."""
        results: List[float] = []
        for params in param_sets:
            exp = self.circuit.run(np.array(params))
            value = float(exp[0])
            if self.noise_shots is not None:
                value += self.rng.normal(0, 1 / np.sqrt(self.noise_shots))
            results.append(value)
        return results

# --------------------------------------------------------------------------- #
#  Unified hybrid layer
# --------------------------------------------------------------------------- #
class UnifiedHybridLayer(nn.Module):
    """A hybrid dense‑to‑quantum layer with batched evaluation and optional noise.

    Parameters
    ----------
    n_features : int
        Number of input features for the classical dense block.
    n_qubits : int
        Number of qubits in the quantum circuit.
    backend : qiskit.providers.backend.Backend, optional
        Backend to run the quantum circuit.  Defaults to Aer qasm simulator.
    shots : int
        Number of shots for the quantum simulation.
    noise_shots : int, optional
        If provided, Gaussian noise with variance 1/shots is added to the
        expectation value to emulate shot noise.
    seed : int, optional
        Random seed for the noise generator.
    """
    def __init__(
        self,
        n_features: int = 1,
        n_qubits: int = 1,
        backend=None,
        shots: int = 100,
        noise_shots: int | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.dense = _DenseBlock(n_features)
        self.quantum = _QuantumCircuitWrapper(n_qubits, backend, shots)
        self.estimator = _Estimator(self.quantum, noise_shots, seed)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Compute the quantum expectation for each input in the batch."""
        dense_out = self.dense(inputs).squeeze(-1).cpu().numpy()
        # Allow dense_out to be 1‑D or 2‑D batch
        param_sets = dense_out if dense_out.ndim == 2 else dense_out.reshape(-1, 1)
        expectations = self.estimator.evaluate(param_sets)
        return torch.tensor(expectations, dtype=torch.float32, device=inputs.device)

    def evaluate_batch(self, param_sets: Sequence[Sequence[float]]) -> List[float]:
        """Expose the estimator for external use."""
        return self.estimator.evaluate(param_sets)

__all__ = ["UnifiedHybridLayer"]
