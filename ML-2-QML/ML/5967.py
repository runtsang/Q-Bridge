import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvFilter(nn.Module):
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def run(self, data) -> float:
        tensor = torch.as_tensor(data, dtype=torch.float32)
        tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean().item()

class SamplerModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(inputs), dim=-1)

class HybridSamplerConv(nn.Module):
    """
    Classical hybrid model that embeds a convolutional filter into a sampler network.
    """
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.conv_filter = ConvFilter(kernel_size, threshold)
        self.sampler = SamplerModule()

    def forward(self, inputs: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Tensor of shape (batch, 2) representing sampler inputs.
            data: Tensor of shape (batch, kernel_size, kernel_size) for convolution.
        Returns:
            Tensor of shape (batch, 2) after sampler softmax.
        """
        conv_outputs = torch.tensor([self.conv_filter.run(d) for d in data],
                                    dtype=inputs.dtype, device=inputs.device)
        conv_outputs = conv_outputs.view(-1, 1)
        combined = torch.cat([inputs, conv_outputs], dim=-1)
        return self.sampler(combined)

__all__ = ["HybridSamplerConv"]
