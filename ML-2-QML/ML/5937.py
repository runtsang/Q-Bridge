import torch
from torch import nn
from Conv import Conv
from.qml import EstimatorQNN as QuantumEstimator

class EstimatorQNN(nn.Module):
    """Hybrid classical‑quantum regressor combining a classical convolution filter with a quantum
    variational circuit. The convolution extracts spatial features from the input data, which
    are then encoded as rotation angles in a 1‑qubit quantum layer that produces the final
    regression output."""
    def __init__(self):
        super().__init__()
        self.conv = Conv()
        self.quantum = QuantumEstimator()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: batch of 2D arrays of shape (kernel_size, kernel_size)
        conv_outputs = []
        for sample in inputs:
            # Convert torch tensor to numpy array
            conv_out = self.conv.run(sample.cpu().numpy())
            conv_outputs.append(conv_out)
        conv_tensor = torch.tensor(conv_outputs, dtype=torch.float32).unsqueeze(1)
        return self.quantum(conv_tensor)

__all__ = ["EstimatorQNN"]
