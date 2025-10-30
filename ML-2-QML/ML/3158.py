import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HybridSampler(nn.Module):
    """
    Classical hybrid sampler that emulates a quantum sampler network.
    Combines a feed‑forward network (derived from SamplerQNN) to produce
    rotation angles for a parametric 2‑qubit circuit (inspired by FCL)
    and simulates the resulting measurement distribution with a deterministic
    NumPy backend.
    """

    def __init__(self, weight_dim: int = 4, seed: int | None = None) -> None:
        super().__init__()
        # 2 → 8 → weight_dim network to map input to rotation angles
        self.param_net = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, weight_dim),
        )
        if seed is not None:
            torch.manual_seed(seed)

    def _quantum_sim(self, inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Analytic simulation of the 2‑qubit circuit:
            ry(inputs[0]) on qubit 0
            ry(inputs[1]) on qubit 1
            CX(0,1)
            ry(weights[0]) on qubit 0
            ry(weights[1]) on qubit 1
            CX(0,1)
            ry(weights[2]) on qubit 0
            ry(weights[3]) on qubit 1
        Returns a 2‑dim probability vector (qubit‑0 = 0 vs 1).
        """
        inp = inputs.detach().cpu().numpy()
        w = weights.detach().cpu().numpy()

        # State vector |00>
        state = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)

        def ry(theta, qubit):
            c = np.cos(theta / 2)
            s = np.sin(theta / 2)
            return np.array([[c, -s], [s, c]])

        def apply_rot(state, theta, qubit):
            R = ry(theta, qubit)
            if qubit == 0:
                state = np.kron(R, np.eye(2)) @ state
            else:
                state = np.kron(np.eye(2), R) @ state
            return state

        cx = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 0, 1],
                       [0, 0, 1, 0]], dtype=complex)

        for theta, qubit in zip(inp, [0, 1]):
            state = apply_rot(state, theta, qubit)
        state = cx @ state
        for theta, qubit in zip(w[:2], [0, 1]):
            state = apply_rot(state, theta, qubit)
        state = cx @ state
        for theta, qubit in zip(w[2:], [0, 1]):
            state = apply_rot(state, theta, qubit)

        probs = np.abs(state) ** 2
        p0 = probs[0] + probs[2]  # qubit 0 = 0
        p1 = probs[1] + probs[3]  # qubit 0 = 1
        probs_2d = np.array([p0, p1], dtype=np.float32)
        return torch.from_numpy(probs_2d)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: inputs shape (batch, 2) → probs shape (batch, 2).
        """
        weights = self.param_net(inputs)
        probs = torch.stack([self._quantum_sim(inp, w) for inp, w in zip(inputs, weights)])
        return F.softmax(probs, dim=-1)

__all__ = ["HybridSampler"]
