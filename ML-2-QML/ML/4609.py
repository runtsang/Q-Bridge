import torch
import torch.nn as nn
import numpy as np

from EstimatorQNN import EstimatorQNN
from SamplerQNN import SamplerQNN
from SelfAttention import SelfAttention

class HybridEstimatorQNN(nn.Module):
    """
    Hybrid estimator that combines a classical feed‑forward regression network, a
    quantum‑inspired self‑attention mechanism, and a quantum sampler.  The
    architecture is inspired by the EstimatorQNN, SelfAttention and SamplerQNN
    reference pairs.
    """

    def __init__(self, quantum_estimator=None):
        super().__init__()
        # Classical regression backbone
        self.classical_net = EstimatorQNN()
        # Quantum sampler producing a probability distribution
        self.sampler = SamplerQNN()
        # Classical self‑attention block with learnable parameters
        self.attention = SelfAttention()
        # Learnable attention parameters (rotation and entangle)
        self.rot_params = nn.Parameter(torch.randn(12))
        self.ent_params = nn.Parameter(torch.randn(3))
        # Optional quantum post‑processing
        self.quantum_estimator = quantum_estimator

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        inputs : torch.Tensor
            Input features of shape (batch, 2).

        Returns
        -------
        torch.Tensor
            Regression predictions, optionally processed by the quantum estimator.
        """
        # 1. Sample probability distribution from input
        probs = self.sampler(inputs)  # (batch, 2)
        # 2. Apply self‑attention to the sampled distribution
        attn_out = self.attention.run(
            rotation_params=self.rot_params.detach().numpy(),
            entangle_params=self.ent_params.detach().numpy(),
            inputs=probs.detach().numpy()
        )
        attn_tensor = torch.tensor(attn_out, dtype=torch.float32, device=inputs.device)
        # 3. Classical regression on the attention output
        reg_out = self.classical_net(attn_tensor)
        # 4. Optional quantum post‑processing
        if self.quantum_estimator is not None:
            # Convert to numpy for quantum estimator
            q_in = reg_out.detach().cpu().numpy()
            q_out = self.quantum_estimator.run(q_in)
            return torch.tensor(q_out, dtype=torch.float32, device=inputs.device)
        return reg_out

__all__ = ["HybridEstimatorQNN"]
