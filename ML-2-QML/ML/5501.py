import torch
import torch.nn as nn
import torch.nn.functional as F


def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Squared overlap of two vectors, normalised."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)


def fidelity_adjacency(
    states: torch.Tensor,
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> torch.Tensor:
    """
    Build a weighted adjacency matrix from pairwise state fidelities.
    Edges with fidelity >= threshold get weight 1.0, secondary values get
    secondary_weight if provided.
    """
    n = states.shape[0]
    adj = torch.zeros((n, n), dtype=torch.float32, device=states.device)
    for i in range(n):
        for j in range(i + 1, n):
            fid = state_fidelity(states[i], states[j])
            if fid >= threshold:
                adj[i, j] = adj[j, i] = 1.0
            elif secondary is not None and fid >= secondary:
                adj[i, j] = adj[j, i] = secondary_weight
    return adj


class KernalAnsatz(nn.Module):
    """Radial‑basis function kernel ansatz."""

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


class Kernel(nn.Module):
    """Wraps :class:`KernalAnsatz` for a single kernel value."""

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()


class HybridFunction(torch.autograd.Function):
    """Differentiable sigmoid activation mimicking a quantum expectation head."""

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float) -> torch.Tensor:  # type: ignore[override]
        outputs = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        (outputs,) = ctx.saved_tensors
        grad_inputs = grad_output * outputs * (1 - outputs)
        return grad_inputs, None


class HybridQuantumNAT(nn.Module):
    """
    Classical hybrid model:
      * CNN + FC projection to 4‑dimensional features.
      * Fidelity‑based adjacency weighting of features.
      * RBF kernel similarity between weighted and unweighted features.
      * Differentiable sigmoid head producing a two‑class probability.
    """

    def __init__(self, shift: float = 0.0) -> None:
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
        # Projection
        self.fc = nn.Sequential(nn.Linear(16 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, 4))
        self.norm = nn.BatchNorm1d(4)
        # Kernel
        self.kernel = Kernel(gamma=1.0)
        self.shift = shift
        self.hybrid = HybridFunction.apply

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.features(x)
        flat = features.view(features.size(0), -1)
        proj = self.fc(flat)
        proj = self.norm(proj)

        # Build adjacency from feature fidelities
        adj = fidelity_adjacency(proj, threshold=0.9)
        weighted_proj = torch.mm(adj, proj)

        # Kernel similarity between weighted and unweighted features
        k_val = self.kernel.forward(proj, weighted_proj).unsqueeze(1)

        # Concatenate features and kernel value
        combined = torch.cat([proj, weighted_proj, k_val], dim=1)
        logits = combined.sum(dim=1, keepdim=True)
        probs = self.hybrid(logits, self.shift)
        return torch.cat([probs, 1 - probs], dim=-1)


__all__ = ["HybridQuantumNAT"]
