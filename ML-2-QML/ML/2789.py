import torch
from torch import nn
import torch.nn.functional as F

class EstimatorQNN(nn.Module):
    """
    Hybrid estimator‑sampler network.

    * Regression head: outputs a scalar prediction.
    * Sampling head: outputs a probability distribution over two classes.
    * Optional quantum delegation: when ``use_quantum=True`` the class forwards
      calls to a wrapped quantum EstimatorQNN/SamplerQNN instance.
    """
    def __init__(self, use_quantum: bool = False, quantum_module: object | None = None) -> None:
        super().__init__()
        self.use_quantum = use_quantum
        self.quantum_module = quantum_module

        self.shared = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 8),
            nn.Tanh()
        )
        self.reg_head = nn.Linear(8, 1)   # regression output
        self.samp_head = nn.Linear(8, 2)  # classification output

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Regression prediction.
        """
        if self.use_quantum and self.quantum_module:
            return self.quantum_module.predict(inputs)
        x = self.shared(inputs)
        return self.reg_head(x)

    def sample(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Sampling probability distribution over two outcomes.
        """
        if self.use_quantum and self.quantum_module:
            return self.quantum_module.sample(inputs)
        x = self.shared(inputs)
        logits = self.samp_head(x)
        return F.softmax(logits, dim=-1)

__all__ = ["EstimatorQNN"]
