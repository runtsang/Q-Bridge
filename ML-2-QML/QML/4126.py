"""Quantum‑augmented classifier that integrates EstimatorQNN and a fully‑connected
quantum layer (FCL). The backbone is identical to the classical version, while
the head can be switched between a parameterised quantum expectation (EstimatorQNN)
or a simple 1‑qubit FCL. This design exposes a gradient‑friendly interface
for end‑to‑end training using Qiskit’s state‑vector estimator.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import qiskit
from qiskit import Aer, transpile, assemble
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import Estimator as StatevectorEstimator


def _build_estimator_qnn(backend):
    """Return an EstimatorQNN instance configured for a single‑qubit circuit."""
    params = [Parameter("input"), Parameter("weight")]
    qc = qiskit.QuantumCircuit(1)
    qc.h(0)
    qc.ry(params[0], 0)
    qc.rx(params[1], 0)
    observable = SparsePauliOp.from_list([("Y", 1)])
    estimator = StatevectorEstimator()
    return EstimatorQNN(
        circuit=qc,
        observables=observable,
        input_params=[params[0]],
        weight_params=[params[1]],
        estimator=estimator
    )


class QuantumFCL:
    """Fully‑connected quantum layer that implements a single‑qubit
    parameterised circuit and returns the expectation of Y."""
    def __init__(self, backend, shots):
        self.backend = backend
        self.shots = shots
        self.circuit = qiskit.QuantumCircuit(1)
        self.theta = Parameter("theta")
        self.circuit.h(0)
        self.circuit.ry(self.theta, 0)
        self.circuit.measure_all()

    def run(self, thetas):
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        expectation = 0.0
        for state, count in result.items():
            prob = count / self.shots
            y = (-1) ** int(state)
            expectation += y * prob
        return np.array([expectation])


class HybridQuantumHead(nn.Module):
    """Hybrid head that can be either EstimatorQNN or QuantumFCL."""
    def __init__(self, mode: str = "estimator", backend=Aer.get_backend("aer_simulator"), shots=100):
        super().__init__()
        self.mode = mode
        if mode == "estimator":
            self.estimator_qnn = _build_estimator_qnn(backend)
        elif mode == "fcl":
            self.fcl = QuantumFCL(backend, shots)
        else:
            raise ValueError(f"Unsupported mode {mode}")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.mode == "estimator":
            probs = self.estimator_qnn.predict(inputs.cpu().numpy())
            return torch.tensor(probs, dtype=inputs.dtype, device=inputs.device)
        else:  # fcl
            probs = self.fcl.run(inputs.cpu().numpy())
            return torch.tensor(probs, dtype=inputs.dtype, device=inputs.device)


class QCNet(nn.Module):
    """Quantum‑augmented CNN for binary classification."""
    def __init__(self, head_mode: str = "estimator", backend=Aer.get_backend("aer_simulator"), shots=100):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.head = HybridQuantumHead(head_mode, backend, shots)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(inputs))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        probs = self.head(x)
        return torch.cat((probs, 1 - probs), dim=-1)


__all__ = ["QCNet", "HybridQuantumHead", "QuantumFCL"]
