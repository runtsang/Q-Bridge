import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridFunction(torch.autograd.Function):
    """Classical sigmoid activation mimicking a quantum expectation head."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float = 0.0) -> torch.Tensor:
        outputs = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (outputs,) = ctx.saved_tensors
        grad_inputs = grad_output * outputs * (1 - outputs)
        return grad_inputs, None

class Hybrid(nn.Module):
    """Dense head that replaces a quantum circuit."""
    def __init__(self, in_features: int, shift: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = inputs.view(inputs.size(0), -1)
        return HybridFunction.apply(self.linear(logits), self.shift)

class SamplerModule(nn.Module):
    """Simple MLP that approximates a sampler QNN."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 2))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(inputs), dim=-1)

class QuantumNATHybrid(nn.Module):
    """Classical approximation of the hybrid Quantum‑NAT model."""
    def __init__(self):
        super().__init__()
        # Feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Fully‑connected projection
        self.fc = nn.Sequential(nn.Linear(16 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, 4))
        self.norm = nn.BatchNorm1d(4)

        # Classical quantum‑approximation layer
        self.quantum_approx = nn.Sequential(
            nn.Linear(4, 64), nn.SiLU(), nn.Linear(64, 4)
        )
        self.q_norm = nn.BatchNorm1d(4)

        # Classification head
        self.classifier = Hybrid(4, shift=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        features = self.features(x)
        flattened = features.view(bsz, -1)
        out = self.fc(flattened)
        out = self.norm(out)

        q_out = self.quantum_approx(out)
        q_out = self.q_norm(q_out)

        logits = self.classifier(q_out)
        probs = torch.sigmoid(logits)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["HybridFunction", "Hybrid", "SamplerModule", "QuantumNATHybrid"]
