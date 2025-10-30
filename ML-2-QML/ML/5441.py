import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QuantumSimulator:
    """Classical simulator of a simple 2-qubit parameterised circuit."""
    def __init__(self, n_qubits=2):
        self.n_qubits = n_qubits

    def expectation(self, params):
        # Dummy analytic expectation: product of cosines
        return np.prod(np.cos(params), axis=-1, keepdims=True)

    def gradient(self, params, shift=np.pi / 2):
        grads = []
        for i in range(params.shape[-1]):
            plus = params.copy()
            minus = params.copy()
            plus[..., i] += shift
            minus[..., i] -= shift
            exp_plus = self.expectation(plus)
            exp_minus = self.expectation(minus)
            grads.append((exp_plus - exp_minus) / (2 * shift))
        return np.stack(grads, axis=-1)

class HybridFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, circuit, shift):
        ctx.shift = shift
        ctx.circuit = circuit
        exp = circuit.expectation(inputs.cpu().numpy())
        result = torch.tensor(exp, device=inputs.device, dtype=inputs.dtype)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        inputs, _ = ctx.saved_tensors
        shift = ctx.shift
        circuit = ctx.circuit
        grads = circuit.gradient(inputs.cpu().numpy(), shift)
        grads = torch.tensor(grads, device=inputs.device, dtype=inputs.dtype)
        return grads * grad_output, None, None

class Hybrid(nn.Module):
    def __init__(self, n_qubits=2, shift=np.pi / 2):
        super().__init__()
        self.circuit = QuantumSimulator(n_qubits)
        self.shift = shift

    def forward(self, inputs):
        return HybridFunction.apply(inputs, self.circuit, self.shift)

def fidelity_adjacency(features, threshold=0.8, secondary=None, secondary_weight=0.5):
    n = features.shape[0]
    adj = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            fid = np.dot(features[i], features[j]) / (np.linalg.norm(features[i]) * np.linalg.norm(features[j]) + 1e-12)
            if fid >= threshold:
                adj[i, j] = adj[j, i] = 1.0
            elif secondary is not None and fid >= secondary:
                adj[i, j] = adj[j, i] = secondary_weight
    return adj

class HybridBinaryClassifier(nn.Module):
    """Classical hybrid binary classifier integrating CNN, graph features, and quantum head."""
    def __init__(self, num_qubits=2, shift=np.pi / 2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_qubits)
        self.hybrid = Hybrid(num_qubits, shift)

    def forward(self, x):
        x = F.relu(self.conv1(x))
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
        q_out = self.hybrid(x)
        probs = torch.sigmoid(q_out).squeeze(-1)
        return torch.stack([probs, 1 - probs], dim=-1)

__all__ = ["HybridBinaryClassifier", "HybridFunction", "Hybrid", "QuantumSimulator", "fidelity_adjacency"]
