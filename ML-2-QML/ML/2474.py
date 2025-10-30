from __future__ import annotations

import torch
from torch import nn
import numpy as np
from qiskit import Aer, execute
from qiskit.circuit import Parameter
from qiskit.circuit import QuantumCircuit


class QuantumFullyConnectedLayer(nn.Module):
    """
    Classical linear layer that produces a quantum parameter and evaluates a
    simple 1‑qubit parameterised circuit. The expectation value of the
    measurement is returned as a scalar tensor.
    """

    def __init__(self, n_features: int = 1, shots: int = 1024) -> None:
        super().__init__()
        self.n_features = n_features
        self.shots = shots
        # Linear transform that turns the input vector into a single parameter
        self.param_linear = nn.Linear(n_features, 1)
        # Build a minimal quantum circuit
        self.qc = self._build_qc()

    def _build_qc(self) -> QuantumCircuit:
        qc = QuantumCircuit(1)
        theta = qc._parameter("theta")
        qc.h(0)
        qc.ry(theta, 0)
        return qc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Produce a parameter for the quantum circuit
        theta = self.param_linear(x).squeeze(-1)
        # Evaluate the quantum circuit on the local Aer simulator
        backend = Aer.get_backend("qasm_simulator")
        param_binds = [{self.qc.parameters[0]: t.item()} for t in theta]
        job = execute(self.qc, backend, shots=self.shots, parameter_binds=param_binds)
        result = job.result()
        counts = result.get_counts(self.qc)
        probs = np.array(list(counts.values())) / self.shots
        states = np.array([int(k, 2) for k in counts.keys()], dtype=float)
        expectation = np.sum(states * probs)
        return torch.tensor(expectation, dtype=x.dtype)


class HybridEstimatorQNN(nn.Module):
    """
    Classical neural network that replaces the second linear layer of the
    original EstimatorQNN with a quantum fully‑connected layer. The network
    can be trained with standard back‑propagation because the quantum layer
    is wrapped in a differentiable PyTorch module that forwards the
    expectation value as a scalar.
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(2, 8)
        self.tanh = nn.Tanh()
        self.quantum_layer = QuantumFullyConnectedLayer(n_features=8)
        self.fc2 = nn.Linear(1, 4)
        self.fc3 = nn.Linear(4, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.tanh(self.fc1(inputs))
        qout = self.quantum_layer(x)
        x = self.tanh(self.fc2(qout))
        return self.fc3(x)


__all__ = ["HybridEstimatorQNN"]
