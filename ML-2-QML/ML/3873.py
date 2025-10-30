import numpy as np
import torch
import torch.nn as nn
from typing import Iterable, Union

class HybridFCL(nn.Module):
    """
    Hybrid fully‑connected layer that can operate in classical or quantum mode.
    Classical mode is a linear layer followed by tanh.
    Quantum mode simulates a parameterised quantum circuit with a classical surrogate
    (cosine of rotated angles) and a trainable linear mapping.
    """
    def __init__(self, n_features: int = 1, hidden_dim: int = 1, n_qubits: int = 0, use_quantum: bool = False):
        super().__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.use_quantum = use_quantum

        if use_quantum:
            if n_qubits <= 0:
                raise ValueError("n_qubits must be > 0 when use_quantum=True")
            self.n_qubits = n_qubits
            # Map input features to qubit parameters
            self.input_to_qubits = nn.Linear(n_features, n_qubits, bias=False)
            # Classical surrogate for the quantum circuit: a trainable weight matrix
            self.quantum_weight = nn.Parameter(torch.randn(n_qubits, n_qubits))
            # Final projection to hidden dimension
            self.out_linear = nn.Linear(n_qubits, hidden_dim)
        else:
            self.linear = nn.Linear(n_features, hidden_dim)
            self.act = nn.Tanh()

    def forward(self, thetas: torch.Tensor) -> torch.Tensor:
        """
        Forward pass. ``thetas`` should be a tensor of shape (batch, n_features).
        """
        if self.use_quantum:
            # Map features to qubit parameters
            x = self.input_to_qubits(thetas)  # (batch, n_qubits)
            # Classical surrogate of quantum expectation: cos of rotated angles
            expectation = torch.cos(x)  # (batch, n_qubits)
            # Optionally add a trainable mixing via quantum_weight
            mixed = torch.matmul(expectation, self.quantum_weight)  # (batch, n_qubits)
            out = self.out_linear(mixed)  # (batch, hidden_dim)
            return out
        else:
            out = self.linear(thetas)
            return self.act(out)

    def run(self, thetas: Union[Iterable[float], np.ndarray]) -> np.ndarray:
        """
        Convenience wrapper matching the original FCL API:
        accepts an iterable or numpy array of input values and returns a numpy array.
        """
        if isinstance(thetas, (list, tuple)):
            thetas = np.array(thetas, dtype=np.float32)
        if isinstance(thetas, np.ndarray):
            thetas = torch.as_tensor(thetas, dtype=torch.float32)
        if thetas.ndim == 1:
            thetas = thetas.unsqueeze(0)
        output = self.forward(thetas)
        return output.detach().numpy()

__all__ = ["HybridFCL"]
