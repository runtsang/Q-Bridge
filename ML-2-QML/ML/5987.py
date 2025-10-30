"""Unified hybrid fully‑connected layer for classical training."""

from __future__ import annotations

from typing import Iterable, List

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class _BaseFCL(nn.Module):
    """Base class that implements the original FCL logic but with a
    more flexible linear mapping and a tunable activation.
    This class can be used directly for purely classical experiments
    or as the head of a hybrid network.
    """
    def __init__(self, n_features: int = 1, bias: bool = True, activation: str = "tanh") -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1, bias=bias)
        self.activation = activation

    def _apply_activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "tanh":
            return torch.tanh(x)
        if self.activation == "sigmoid":
            return torch.sigmoid(x)
        if self.activation == "relu":
            return torch.relu(x)
        raise ValueError(f"Unsupported activation: {self.activation}")

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Run a batched forward pass and return a NumPy array."""
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        out = self.linear(values)
        out = self._apply_activation(out)
        return out.mean(dim=0).detach().numpy()


class UnifiedHybridFC(nn.Module):
    """Hybrid fully‑connected layer that can be composed with a CNN
    back‑bone.  The class exposes two modes:

    1. Classical mode – uses _BaseFCL to compute a deterministic output.
    2. Quantum mode – forwards the activations through a parameterised
       quantum circuit and returns the expectation value.
    """
    def __init__(
        self,
        n_features: int = 1,
        n_qubits: int = 1,
        backend=None,
        shots: int = 100,
        shift: float = np.pi / 2,
        bias: bool = True,
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots
        self.shift = shift
        self.classical_head = _BaseFCL(n_features, bias, activation)

        # Quantum circuit only instantiated if backend is supplied
        if backend is not None:
            self._quantum_circuit = _QuantumCircuit(
                n_qubits, backend, shots, shift
            )
        else:
            self._quantum_circuit = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass that automatically selects the appropriate mode."""
        if self._quantum_circuit is None:
            # Classical mode
            return self.classical_head.run(inputs.tolist())
        # Quantum mode
        return self._quantum_circuit.run(inputs.detach().cpu().numpy())

    def set_quantum_mode(self, enable: bool) -> None:
        """Toggle quantum mode on or off."""
        if enable and self._quantum_circuit is None:
            raise RuntimeError("Quantum backend not configured.")
        self._quantum_circuit = None if not enable else self._quantum_circuit

    def __repr__(self) -> str:
        mode = "Quantum" if self._quantum_circuit else "Classical"
        return f"<UnifiedHybridFC mode={mode} n_features={self.classical_head.linear.in_features}>"

class _QuantumCircuit:
    """Thin wrapper around a Qiskit parameterised circuit that
    supports batched execution and a simple expectation calculation.
    """
    def __init__(self, n_qubits: int, backend, shots: int, shift: float):
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots
        self.shift = shift

        # Build a simple parametric circuit
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        self.theta = qiskit.circuit.Parameter("theta")
        self.circuit.h(range(n_qubits))
        self.circuit.barrier()
        self.circuit.ry(self.theta, range(n_qubits))
        self.circuit.measure_all()

    def run(self, thetas: np.ndarray) -> torch.Tensor:
        """Execute the circuit for each input in the batch and return
        a tensor of expectation values.
        """
        if thetas.ndim == 1:
            thetas = thetas.reshape(-1, 1)
        results = []
        for theta in thetas:
            job = qiskit.execute(
                self.circuit,
                self.backend,
                shots=self.shots,
                parameter_binds=[{self.theta: theta[0]}],
            )
            counts = job.result().get_counts(self.circuit)
            probs = np.array(list(counts.values())) / self.shots
            states = np.array(list(counts.keys()), dtype=float)
            exp_val = np.sum(states * probs)
            results.append(exp_val)
        return torch.tensor(results, dtype=torch.float32).unsqueeze(-1)

__all__ = ["UnifiedHybridFC"]
