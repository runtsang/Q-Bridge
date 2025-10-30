"""Advanced quantum‑classical hybrid binary classifier.

This module implements a hybrid network that augments the classical
backbone with a per‑feature two‑qubit quantum feature‑map.  Each
feature of the 84‑dimensional vector is fed into its own parameterized
circuit; the expectation value of Pauli‑Z on qubit 0 is used as the
feature for the final linear classifier.  The quantum head also
provides the variance of each expectation for richer supervision.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import qiskit
from qiskit import assemble, transpile
from qiskit.providers.aer import AerSimulator


class ResidualBlock(nn.Module):
    """Residual block with two convolutional layers and a skip connection."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                               stride=stride, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size,
                               stride=stride, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if in_channels!= out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                      stride=stride)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)


class TwoQubitFeatureMap:
    """Parameterised two‑qubit circuit used as a feature map."""
    def __init__(self, backend: AerSimulator, shots: int = 1024):
        self.backend = backend
        self.shots = shots
        self.circuit = qiskit.QuantumCircuit(2)
        self.theta = qiskit.circuit.Parameter("theta")
        # Simple entangling feature map
        self.circuit.h(0)
        self.circuit.h(1)
        self.circuit.rx(self.theta, 0)
        self.circuit.rz(self.theta, 1)
        self.circuit.cx(0, 1)
        self.circuit.measure_all()

    def run(self, theta: float) -> tuple[float, float]:
        """Execute the circuit for a single parameter value.

        Returns the expectation value of Pauli‑Z on qubit 0 and its
        variance.
        """
        compiled = transpile(self.circuit, self.backend)
        bound = compiled.bind_parameters({self.theta: theta})
        qobj = assemble(bound, shots=self.shots)
        result = self.backend.run(qobj).result()
        counts = result.get_counts()
        expectation = 0.0
        for bitstring, count in counts.items():
            prob = count / self.shots
            # In Qiskit the least‑significant bit corresponds to qubit 0
            z = 1.0 if bitstring[-1] == "0" else -1.0
            expectation += prob * z
        variance = 1.0 - expectation ** 2
        return expectation, variance


class QuantumFeatureMapFunction(torch.autograd.Function):
    """Differentiable wrapper that evaluates a vector of feature maps."""
    @staticmethod
    def forward(ctx, features: torch.Tensor, circuit: TwoQubitFeatureMap, shift: float):
        expectations, variances = [], []
        for val in features.tolist():
            exp, var = circuit.run(val)
            expectations.append(exp)
            variances.append(var)
        expectations = torch.tensor(expectations, dtype=torch.float32)
        variances = torch.tensor(variances, dtype=torch.float32)
        ctx.save_for_backward(features, expectations, variances)
        ctx.circuit = circuit
        ctx.shift = shift
        return expectations, variances

    @staticmethod
    def backward(ctx, grad_outputs):
        grad_expectation, grad_variance = grad_outputs
        features, expectations, variances = ctx.saved_tensors
        shift = ctx.shift
        circuit = ctx.circuit
        grads = []
        for val in features.tolist():
            exp_plus = circuit.run(val + shift)[0]
            exp_minus = circuit.run(val - shift)[0]
            grad = (exp_plus - exp_minus) / (2 * shift)
            grads.append(grad)
        grads = torch.tensor(grads, dtype=torch.float32)
        # Gradient w.r.t. variance is ignored (set to zero)
        return grads * grad_expectation, None, None


class QuantumHybrid(nn.Module):
    """Hybrid layer that maps a feature vector to quantum expectations."""
    def __init__(self, n_features: int, backend: AerSimulator, shots: int = 512,
                 shift: float = np.pi / 2):
        super().__init__()
        self.n_features = n_features
        self.circuit = TwoQubitFeatureMap(backend, shots)
        self.shift = shift

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        expectations, variances = QuantumFeatureMapFunction.apply(features, self.circuit, self.shift)
        expectations = expectations.view(-1, self.n_features)
        variances = variances.view(-1, self.n_features)
        return expectations, variances


class AdvancedHybridBinaryClassifier(nn.Module):
    """Hybrid network that augments the classical backbone with a quantum head."""
    def __init__(self) -> None:
        super().__init__()
        # Classical backbone (identical to the ML version)
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.res_block = ResidualBlock(6, 12, kernel_size=3, stride=1)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(12, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 84)  # 84‑dimensional feature vector
        # Quantum hybrid head
        backend = AerSimulator()
        self.quantum = QuantumHybrid(n_features=84, backend=backend, shots=256,
                                     shift=np.pi / 2)
        # Final classifier that consumes both expectation and variance
        self.classifier = nn.Linear(84 * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Classical feature extraction
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = self.res_block(x)
        x = self.drop2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # 84‑dimensional features
        # Quantum hybrid head
        expectation, variance = self.quantum(x)
        # Concatenate expectation and variance
        features = torch.cat([expectation, variance], dim=-1)
        logits = self.classifier(features)
        probs = torch.sigmoid(logits)
        return torch.cat((probs, 1 - probs), dim=-1)


__all__ = ["TwoQubitFeatureMap", "QuantumFeatureMapFunction",
           "QuantumHybrid", "AdvancedHybridBinaryClassifier"]
