import torch
import torch.nn as nn
import torch.nn.functional as F

# Classical helpers from the seed project
from SamplerQNN import SamplerQNN as QuantumSampler
from QCNN import QCNNModel
from FCL import FCL as FullyConnectedLayer

class SamplerQNN__gen093(nn.Module):
    """
    Hybrid classical sampler network that merges a QCNN feature extractor,
    a fully‑connected layer, and a quantum sampler.  The forward pass
    processes the input through the classical stack and then feeds the
    resulting expectation value into the quantum sampler to produce a
    probability distribution.
    """

    def __init__(self) -> None:
        super().__init__()
        # Classical feature extractor (mimics a QCNN)
        self.feature_extractor = QCNNModel()
        # Classical fully‑connected layer (mimics a quantum FCL)
        self.fc_layer = FullyConnectedLayer()
        # Quantum sampler from the seed QML module
        self.quantum_sampler = QuantumSampler()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the classical components.

        Parameters
        ----------
        inputs
            Tensor of shape (batch, 8) compatible with QCNNModel.

        Returns
        -------
        torch.Tensor
            Probability distribution over two outputs (softmax).
        """
        # Extract features
        features = self.feature_extractor(inputs)
        # Flatten features to feed into the fully‑connected layer
        flat = features.view(-1).tolist()
        # Classical expectation value
        expectation = self.fc_layer.run(flat)
        # Feed expectation into the quantum sampler
        probs = self.quantum_sampler(
            torch.tensor([expectation], dtype=torch.float32)
        )
        return probs

    def sample(self, thetas: list[float]) -> torch.Tensor:
        """
        Run the quantum sampler directly with custom parameters.

        Parameters
        ----------
        thetas
            List of parameters for the quantum circuit.

        Returns
        -------
        torch.Tensor
            Sampled probability distribution.
        """
        return self.quantum_sampler(
            torch.tensor(thetas, dtype=torch.float32)
        )

__all__ = ["SamplerQNN__gen093"]
