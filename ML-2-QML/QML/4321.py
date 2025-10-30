"""Hybrid estimator that fuses a PyTorch model, a Qiskit quantum circuit,
and a quantum self‑attention block.

The estimator uses a parameter‑shift rule to make the quantum part
differentiable with respect to the circuit parameters.  Shot noise is
added when `shots` is specified, mirroring the behaviour of the
classical counterpart.

Typical usage
-------------
>>> from torch import nn
>>> model = nn.Linear(10, 1)
>>> quantum = QuantumCircuitWrapper(n_qubits=2, shots=500)
>>> attention = QuantumSelfAttention(n_qubits=4)
>>> est = HybridEstimator(
...     model=model,
...     quantum_circuit=quantum,
...     attention=attention,
...     weights={"classical": 0.5, "quantum": 0.3, "attention": 0.2},
... )
>>> outputs = est.evaluate(
...     observables=[lambda out: out.mean()],
...     parameter_sets=[[0.1]*10, [0.2]*10],
... )
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional, Dict

import numpy as np
import torch
from torch import nn
from torch.autograd import Function

import qiskit
from qiskit import assemble, transpile
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info import Statevector

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


class QuantumCircuitWrapper:
    """Parametrised two‑qubit circuit executed on an Aer backend."""

    def __init__(self, n_qubits: int, shots: int) -> None:
        self.circuit = QuantumCircuit(n_qubits)
        all_qubits = list(range(n_qubits))
        self.theta = Parameter("θ")
        self.circuit.h(all_qubits)
        self.circuit.ry(self.theta, all_qubits)
        self.circuit.measure_all()
        self.backend = qiskit.Aer.get_backend("aer_simulator")
        self.shots = shots

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self.circuit, self.backend)
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
            probs = counts / self.shots
            return np.sum(states * probs)
        if isinstance(result, list):
            return np.array([expectation(item) for item in result])
        return np.array([expectation(result)])


class QuantumSelfAttention:
    """Quantum self‑attention block implemented with Qiskit."""

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.qr = qiskit.QuantumRegister(n_qubits, "q")
        self.cr = qiskit.ClassicalRegister(n_qubits, "c")
        self.backend = qiskit.Aer.get_backend("qasm_simulator")

    def _build_circuit(
        self, rotation_params: np.ndarray, entangle_params: np.ndarray
    ) -> QuantumCircuit:
        circuit = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        circuit.measure(self.qr, self.cr)
        return circuit

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ) -> Dict[str, int]:
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = qiskit.execute(circuit, self.backend, shots=shots)
        return job.result().get_counts(circuit)


class HybridFunction(Function):
    """Differentiable interface between PyTorch and a parametrised circuit."""

    @staticmethod
    def forward(
        ctx,
        inputs: torch.Tensor,
        circuit: QuantumCircuitWrapper,
        shift: float,
    ) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        expectation = ctx.circuit.run(inputs.tolist())
        result = torch.tensor([expectation[0]], dtype=torch.float32)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = np.ones_like(inputs.tolist()) * ctx.shift
        grads = []
        for idx, val in enumerate(inputs.tolist()):
            exp_plus = ctx.circuit.run([val + shift[idx]])[0]
            exp_minus = ctx.circuit.run([val - shift[idx]])[0]
            grads.append(exp_plus - exp_minus)
        grads_tensor = torch.tensor([grads], dtype=torch.float32)
        return grads_tensor * grad_output, None, None


class HybridEstimator:
    """Evaluate a weighted combination of classical, quantum, and attention signals
    with differentiable quantum support.

    The class mirrors the classical estimator but injects a quantum layer that
    can be trained via the parameter‑shift rule.  The attention block may also
    be quantum, providing a fully quantum‑aware pipeline.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        quantum_circuit: QuantumCircuitWrapper | None = None,
        attention: QuantumSelfAttention | None = None,
        weights: Dict[str, float] | None = None,
        shift: float = np.pi / 2,
    ) -> None:
        self.model = model
        self.quantum_circuit = quantum_circuit
        self.attention = attention
        self.weights = weights or {"classical": 1.0, "quantum": 0.0, "attention": 0.0}
        self.shift = shift

    def _compute_components(
        self, params: Sequence[float]
    ) -> torch.Tensor:
        """Return weighted sum of components as a torch tensor."""
        inputs = torch.as_tensor(params, dtype=torch.float32)
        with torch.no_grad():
            classical_out = self.model(inputs.unsqueeze(0)).squeeze()
        classical_out = torch.tensor(classical_out, dtype=torch.float32)

        if self.quantum_circuit is not None:
            quantum_expect = torch.tensor(
                self.quantum_circuit.run(np.asarray(params)), dtype=torch.float32
            )
        else:
            quantum_expect = torch.tensor(0.0, dtype=torch.float32)

        if self.attention is not None:
            # Random rotation/entangle params for demo; replace with learnable ones
            rot = np.random.rand(12)
            ent = np.random.rand(3)
            att_counts = self.attention.run(rot, ent)
            # Convert counts to expectation value (simple average of binary strings)
            total = sum(att_counts.values())
            exp = sum(int(k, 2) * v for k, v in att_counts.items()) / total
            attention_out = torch.tensor(exp, dtype=torch.float32)
        else:
            attention_out = torch.tensor(0.0, dtype=torch.float32)

        weighted = (
            self.weights.get("classical", 0.0) * classical_out
            + self.weights.get("quantum", 0.0) * quantum_expect
            + self.weights.get("attention", 0.0) * attention_out
        )
        return weighted

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        observables = list(observables) or [lambda out: out.mean()]
        results: List[List[float]] = []

        for params in parameter_sets:
            combined = self._compute_components(params)
            row: List[float] = []
            for obs in observables:
                value = obs(combined)
                if isinstance(value, torch.Tensor):
                    scalar = float(value.mean().cpu())
                else:
                    scalar = float(value)
                row.append(scalar)
            results.append(row)

        if shots is not None:
            rng = np.random.default_rng(seed)
            noisy: List[List[float]] = []
            for row in results:
                noisy_row = [
                    float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row
                ]
                noisy.append(noisy_row)
            return noisy

        return results


__all__ = ["HybridEstimator", "QuantumCircuitWrapper", "QuantumSelfAttention"]
