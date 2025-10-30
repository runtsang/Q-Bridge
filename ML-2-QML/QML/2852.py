"""Quantum classifier that mirrors the classical interface.

The UnifiedClassifier class builds a data‑uploading variational ansatz
with RX/RY/CZ gates and a parameter‑shift autograd function for
back‑propagation.  The static method build_classifier_circuit returns
the circuit, encoding, weights and observables.  The forward pass
accepts a batch of parameter vectors and outputs the expectation of
Z‑observables for each qubit.
"""

from __future__ import annotations

import numpy as np
from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import qiskit
from qiskit import QuantumCircuit, transpile, assemble
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

__all__ = ["UnifiedClassifier"]

class HybridFunction(torch.autograd.Function):
    """Differentiable interface that runs the quantum circuit and
    evaluates the expectation of the supplied observables.
    """

    @staticmethod
    def forward(
        ctx,
        inputs: torch.Tensor,
        circuit: QuantumCircuit,
        encoding: Iterable[ParameterVector],
        weights: Iterable[ParameterVector],
        observables: list[SparsePauliOp],
        backend,
        shots: int,
        shift: float,
    ) -> torch.Tensor:
        batch, num_params = inputs.shape
        num_encoding = len(encoding)
        num_weights = len(weights)
        expectations = torch.zeros((batch, len(observables)), dtype=torch.float32, device=inputs.device)

        for i in range(batch):
            param_bind = {}
            for j, p in enumerate(encoding):
                param_bind[p] = inputs[i, j].item()
            for j, p in enumerate(weights):
                param_bind[p] = inputs[i, num_encoding + j].item()

            compiled = transpile(circuit, backend)
            qobj = assemble(compiled, parameter_binds=[param_bind], shots=shots)
            job = backend.run(qobj)
            result = job.result()
            counts = result.get_counts()

            for k, obs in enumerate(observables):
                exp = 0.0
                for state, cnt in counts.items():
                    exp += obs.expectation_value(state, dtype=np.float64) * (cnt / shots)
                expectations[i, k] = exp

        ctx.save_for_backward(inputs, expectations)
        ctx.circuit = circuit
        ctx.encoding = encoding
        ctx.weights = weights
        ctx.observables = observables
        ctx.backend = backend
        ctx.shots = shots
        ctx.shift = shift
        return expectations

    @staticmethod
    def backward(ctx, grad_output):
        inputs, _ = ctx.saved_tensors
        circuit = ctx.circuit
        encoding = ctx.encoding
        weights = ctx.weights
        observables = ctx.observables
        backend = ctx.backend
        shots = ctx.shots
        shift = ctx.shift

        batch, num_params = inputs.shape
        num_encoding = len(encoding)
        grad_inputs = torch.zeros_like(inputs)

        # Parameter‑shift rule for each parameter
        for i in range(batch):
            for p_idx in range(num_params):
                # +shift
                param_bind_plus = {}
                for j, p in enumerate(encoding):
                    param_bind_plus[p] = inputs[i, j].item() if j!= p_idx else inputs[i, j].item() + shift
                for j, p in enumerate(weights):
                    param_bind_plus[p] = inputs[i, num_encoding + j].item()

                compiled = transpile(circuit, backend)
                qobj = assemble(compiled, parameter_binds=[param_bind_plus], shots=shots)
                job = backend.run(qobj)
                result = job.result()
                counts_plus = result.get_counts()
                exp_plus = 0.0
                for obs in observables:
                    exp = 0.0
                    for state, cnt in counts_plus.items():
                        exp += obs.expectation_value(state, dtype=np.float64) * (cnt / shots)
                    exp_plus += exp

                # -shift
                param_bind_minus = {}
                for j, p in enumerate(encoding):
                    param_bind_minus[p] = inputs[i, j].item() if j!= p_idx else inputs[i, j].item() - shift
                for j, p in enumerate(weights):
                    param_bind_minus[p] = inputs[i, num_encoding + j].item()

                compiled = transpile(circuit, backend)
                qobj = assemble(compiled, parameter_binds=[param_bind_minus], shots=shots)
                job = backend.run(qobj)
                result = job.result()
                counts_minus = result.get_counts()
                exp_minus = 0.0
                for obs in observables:
                    exp = 0.0
                    for state, cnt in counts_minus.items():
                        exp += obs.expectation_value(state, dtype=np.float64) * (cnt / shots)
                    exp_minus += exp

                grad = (exp_plus - exp_minus) / (2 * shift)
                # Scale by upstream gradient
                grad_inputs[i, p_idx] = grad * grad_output[i].sum()

        return (
            grad_inputs,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

class HybridLayer(nn.Module):
    """Hybrid layer that forwards activations through a quantum circuit."""

    def __init__(
        self,
        num_qubits: int,
        depth: int,
        backend=None,
        shots: int = 1024,
        shift: float = np.pi / 2,
    ) -> None:
        super().__init__()
        self.num_qubits = num_qubits
        self.depth = depth
        self.backend = backend or qiskit.Aer.get_backend("aer_simulator")
        self.shots = shots
        self.shift = shift

        self.circuit, self.encoding, self.weights, self.observables = self.build_classifier_circuit(
            num_qubits, depth
        )

    @staticmethod
    def build_classifier_circuit(num_qubits: int, depth: int):
        """Construct a data‑uploading ansatz with RX, RY and CZ gates."""
        encoding = ParameterVector("x", num_qubits)
        weights = ParameterVector("theta", num_qubits * depth)

        circuit = QuantumCircuit(num_qubits)

        # Data‑encoding layer
        for qubit, param in enumerate(encoding):
            circuit.rx(param, qubit)

        # Variational layers
        for d in range(depth):
            for qubit, param in enumerate(weights[d * num_qubits : (d + 1) * num_qubits]):
                circuit.ry(param, qubit)
            for qubit in range(num_qubits - 1):
                circuit.cz(qubit, qubit + 1)

        observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
            for i in range(num_qubits)
        ]
        return circuit, list(encoding), list(weights), observables

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run the hybrid layer on a batch of parameter vectors."""
        return HybridFunction.apply(
            inputs,
            self.circuit,
            self.encoding,
            self.weights,
            self.observables,
            self.backend,
            self.shots,
            self.shift,
        )

class UnifiedClassifier(nn.Module):
    """Complete hybrid classifier that combines a classical backbone
    with a quantum expectation head.
    """

    def __init__(self, num_features: int, depth: int, *, shift: float = np.pi / 2):
        super().__init__()
        # Classical backbone producing a vector of parameters for the quantum layer
        self.backbone = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.ReLU(),
            nn.Linear(num_features, num_features),
            nn.ReLU(),
            nn.Linear(num_features, num_features * depth),
        )
        # Quantum head
        self.hybrid = HybridLayer(num_qubits=depth, depth=depth, shift=shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pass through classical backbone
        params = self.backbone(x)
        # Quantum expectation head
        quantum_out = self.hybrid(params)
        # Convert to probabilities (optional)
        probs = torch.sigmoid(quantum_out)
        return torch.cat([probs, 1 - probs], dim=-1)
