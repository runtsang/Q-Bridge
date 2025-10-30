"""UnifiedClassifierHybrid – classical backbone for hybrid classification.

The module aggregates the design patterns from the four reference pairs:
* Feed‑forward residual network (QuantumClassifierModel.py)
* Parameter‑clipped fraud‑detection layers (FraudDetection.py)
* Self‑attention block (SelfAttention.py)
* Fidelity‑based adjacency graph (GraphQNN.py)

The resulting class implements a residual feed‑forward network with optional
self‑attention and a quantum‑derived feature vector that can be supplied
by the companion QML module.

Typical usage::

    from UnifiedClassifierHybrid import UnifiedClassifierHybrid
    model = UnifiedClassifierHybrid(num_features=32, depth=3, attention_dim=4)
    logits, probs = model.predict(X)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# 1. Fraud‑layer parameters (clipped fully‑connected layer)
# --------------------------------------------------------------------------- #
@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

class FraudLayer(nn.Module):
    """A linear layer with optional parameter clipping and a post‑processing
    scaling/shift operation inspired by the photonic fraud‑detection stack."""
    def __init__(self, params: FraudLayerParameters, clip: bool = False) -> None:
        super().__init__()
        weight = torch.tensor(
            [[params.bs_theta, params.bs_phi],
             [params.squeeze_r[0], params.squeeze_r[1]]],
            dtype=torch.float32,
        )
        bias = torch.tensor(params.phases, dtype=torch.float32)
        if clip:
            weight = weight.clamp(-5.0, 5.0)
            bias = bias.clamp(-5.0, 5.0)
        linear = nn.Linear(2, 2, bias=True)
        with torch.no_grad():
            linear.weight.copy_(weight)
            linear.bias.copy_(bias)
        self.linear = linear
        self.activation = nn.Tanh()
        self.register_buffer("scale", torch.tensor(params.displacement_r, dtype=torch.float32))
        self.register_buffer("shift", torch.tensor(params.displacement_phi, dtype=torch.float32))

    def forward(self, inp: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        out = self.activation(self.linear(inp))
        out = out * self.scale + self.shift
        return out

def build_fraud_detection_model(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Return a Sequential model composed of one unclipped fraud layer followed
    by one or more clipped layers and a final linear output."""
    modules: List[nn.Module] = [FraudLayer(input_params, clip=False)]
    modules.extend(FraudLayer(lp, clip=True) for lp in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

# --------------------------------------------------------------------------- #
# 2. Classical self‑attention block
# --------------------------------------------------------------------------- #
class ClassicalSelfAttention(nn.Module):
    """Simple self‑attention module that mimics the block in SelfAttention.py."""
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        scores = F.softmax(q @ k.T / np.sqrt(self.embed_dim), dim=-1)
        return scores @ v

# --------------------------------------------------------------------------- #
# 3. Fidelity‑based adjacency graph (graph‑based post‑processing)
# --------------------------------------------------------------------------- #
def fidelity_graph(
    states: List[torch.Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> np.ndarray:
    """Return a weighted adjacency matrix derived from pairwise fidelities."""
    n = len(states)
    adj = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            a = states[i] / (torch.norm(states[i]) + 1e-12)
            b = states[j] / (torch.norm(states[j]) + 1e-12)
            fid = float(torch.dot(a, b).item() ** 2)
            if fid >= threshold:
                adj[i, j] = adj[j, i] = 1.0
            elif secondary is not None and fid >= secondary:
                adj[i, j] = adj[j, i] = secondary_weight
    return adj

# --------------------------------------------------------------------------- #
# 4. Residual feed‑forward backbone
# --------------------------------------------------------------------------- #
class ResidualBlock(nn.Module):
    """One residual block composed of a linear layer followed by ReLU."""
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)
        self.relu   = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.linear(x))

def build_residual_backbone(num_features: int, depth: int) -> nn.Sequential:
    """Create a sequential residual backbone."""
    layers: List[nn.Module] = []
    for _ in range(depth):
        layers.append(ResidualBlock(num_features))
    layers.append(nn.Linear(num_features, 2))
    return nn.Sequential(*layers)

# --------------------------------------------------------------------------- #
# 5. Unified hybrid classifier
# --------------------------------------------------------------------------- #
class UnifiedClassifierHybrid(nn.Module):
    """Hybrid classical‑quantum classifier that stitches together the
    components defined above.

    The forward pass consists of:
        1. Residual feed‑forward backbone
        2. Optional self‑attention over the first ``attention_dim`` features
        3. Optional quantum embedding (placeholder) that augments the logits
    """
    def __init__(
        self,
        num_features: int,
        depth: int = 3,
        *,
        attention_dim: int = 4,
        quantum: bool = True,
        quantum_depth: int = 2,
    ) -> None:
        super().__init__()
        self.quantum = quantum
        self.backbone = build_residual_backbone(num_features, depth)
        self.attention = ClassicalSelfAttention(attention_dim)
        # Parameters that will be mapped to a quantum circuit
        self.q_params = nn.Parameter(torch.randn(num_features, quantum_depth))

    # ------------------------------------------------------------------ #
    # Forward pass
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits of shape (batch, 2)."""
        # backbone
        logits = self.backbone(x)
        # attention (only on first ``attention_dim`` cols)
        att = self.attention(x[:, :self.attention.embed_dim])
        logits = logits + att
        # quantum feature (placeholder)
        if self.quantum:
            # map input to parameters for the quantum circuit
            q_in = torch.tanh((x @ self.q_params.t()).t())
            # simple surrogate for a quantum expectation:
            quantum_out = torch.mean(2 * torch.sigmoid(-q_in), dim=1, keepdim=True)
            logits = logits + quantum_out
        return logits

    # ------------------------------------------------------------------ #
    # Convenience helpers
    # ------------------------------------------------------------------ #
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return logits and probabilities."""
        logits = self(x)
        probs  = F.softmax(logits, dim=-1)
        return logits, probs

    def softmax(self, logits: torch.Tensor) -> torch.Tensor:
        return F.softmax(logits, dim=-1)

    # ------------------------------------------------------------------ #
    # Training utilities – freeze everything except the final linear head
    # ------------------------------------------------------------------ #
    def freeze_except_head(self) -> None:
        for p in self.parameters():
            p.requires_grad = False
        for p in self.backbone[-1].parameters():
            p.requires_grad = True

# --------------------------------------------------------------------------- #
# 6. Exposed API
# --------------------------------------------------------------------------- #
__all__ = [
    "FraudLayerParameters",
    "FraudLayer",
    "build_fraud_detection_model",
    "ClassicalSelfAttention",
    "fidelity_graph",
    "ResidualBlock",
    "build_residual_backbone",
    "UnifiedClassifierHybrid",
]
