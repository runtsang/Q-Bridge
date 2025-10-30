"""Quantum components for the hybrid classifier.

This module provides a parameterized circuit builder, a wrapper around a
Qiskit Aer simulator, and a differentiable hybrid layer that can be
plugged into a PyTorch model.  The design follows the patterns in the
original seed while adding support for batched inputs and a simple
finite‑difference gradient approximation.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit as QC, assemble, transpile, Aer
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


def build_classifier_circuit(num_qubits: int, depth: int = 2) -> Tuple[QC, Iterable[ParameterVector], Iterable[ParameterVector], List[SparsePauliOp]]:
    """Build a parameterized ansatz with data‑encoding and variational layers.

    The circuit follows the “data‑uploading” style: each input feature is
    encoded with an RX gate, then a stack of RY rotations and CZ
    entangling gates is applied.  The returned tuple contains the
    circuit, the encoding parameters, the variational parameters, and
    a list of observables (Z on each qubit) that are used to compute
    the output expectation values.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QC(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]

    return circuit, list(encoding), list(weights), observables


class QuantumCircuit:
    """Thin wrapper around a Qiskit circuit that can be executed on the Aer simulator.

    The wrapper exposes a ``run`` method that accepts a NumPy array of shape
    (batch_size, num_qubits) and returns the expectation value of the
    observables that were supplied when the circuit was built.
    """
    def __init__(self, circuit: QC, backend=None, shots: int = 1024):
        self._circuit = circuit
        self.backend = backend or Aer.get_backend("aer_simulator")
        self.shots = shots
        self.num_qubits = circuit.num_qubits

    def run(self, inputs: np.ndarray) -> np.ndarray:
        """Execute the circuit for a batch of inputs.

        Parameters
        ----------
        inputs
            Array of shape (batch_size, num_qubits) with the values that
            should be bound to the encoding parameters.

        Returns
        -------
        expectations
            Array of shape (batch_size, num_qubits) containing the
            expectation values of the Z observables for each input.
        """
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)

        compiled = transpile(self._circuit, self.backend)
        batch_size = inputs.shape[0]
        expectations = np.zeros((batch_size, self.num_qubits))

        for i in range(batch_size):
            # Bind encoding parameters
            bind_dict = {self._circuit.parameters[j]: inputs[i, j] for j in range(self.num_qubits)}
            # Bind variational parameters (assumed already bound externally)
            qobj = assemble(compiled, shots=self.shots, parameter_binds=[bind_dict])
            job = self.backend.run(qobj)
            result = job.result()
            counts = result.get_counts()
            # Convert counts to expectation of Z for each qubit
            for q in range(self.num_qubits):
                exp = 0.0
                for state, count in counts.items():
                    # state is a string like '01', bits are reversed in Qiskit
                    bit = int(state[::-1][q])
                    exp += (1 if bit == 0 else -1) * count
                exp /= self.shots
                expectations[i, q] = exp

        return expectations


class HybridFunction(torch.autograd.Function):
    """Differentiable bridge between a PyTorch tensor and a Qiskit circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float = np.pi / 2) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        # Run the circuit on the CPU
        with torch.no_grad():
            numpy_inputs = inputs.detach().cpu().numpy()
            expectations = ctx.circuit.run(numpy_inputs)
        result = torch.tensor(expectations, dtype=inputs.dtype, device=inputs.device)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        # Finite difference approximation
        grad_inputs = torch.zeros_like(inputs)
        for idx in range(inputs.shape[0]):
            # Positive shift
            pos = inputs[idx] + shift
            pos_exp = ctx.circuit.run(pos.detach().cpu().numpy())[0]
            # Negative shift
            neg = inputs[idx] - shift
            neg_exp = ctx.circuit.run(neg.detach().cpu().numpy())[0]
            grad = pos_exp - neg_exp
            grad_inputs[idx] = grad
        return grad_inputs * grad_output, None, None


class Hybrid(nn.Module):
    """Hybrid layer that forwards activations through a quantum circuit."""
    def __init__(self, n_qubits: int, backend=None, shots: int = 1024, shift: float = np.pi / 2):
        super().__init__()
        circuit, _, _, _ = build_classifier_circuit(n_qubits, depth=2)
        self.quantum_circuit = QuantumCircuit(circuit, backend=backend, shots=shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs, self.quantum_circuit, self.shift)


__all__ = ["build_classifier_circuit", "QuantumCircuit", "HybridFunction", "Hybrid"]
