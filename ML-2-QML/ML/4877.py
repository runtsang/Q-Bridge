import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ------------------------------------------------------------------
# Classical convolutional filter (drop‑in replacement for quanvolution)
# ------------------------------------------------------------------
class ConvFilter(nn.Module):
    """2×2 convolution followed by a sigmoid threshold."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def run(self, data: np.ndarray) -> float:
        """Return the mean sigmoid activation for a 2×2 patch."""
        tensor = torch.as_tensor(data, dtype=torch.float32)
        tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean().item()

# ------------------------------------------------------------------
# Classical RBF kernel (placeholder for quantum kernel)
# ------------------------------------------------------------------
class KernalAnsatz(nn.Module):
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

def kernel_matrix(a: list[torch.Tensor], b: list[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    kernel = Kernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# ------------------------------------------------------------------
# Hybrid sampler network
# ------------------------------------------------------------------
class HybridSamplerQNN(nn.Module):
    """
    Classical hybrid sampler that stitches together a convolutional
    feature extractor, an LSTM sequence model, a radial‑basis kernel,
    and a linear classifier.  The architecture mirrors the quantum
    counterpart in the QML module, enabling head‑to‑head comparisons.
    """
    def __init__(self,
                 input_dim: int = 4,
                 hidden_dim: int = 8,
                 n_qubits: int = 0,
                 conv_kernel_size: int = 2,
                 threshold: float = 0.0,
                 gamma: float = 1.0) -> None:
        super().__init__()
        if input_dim!= conv_kernel_size ** 2:
            raise ValueError("For this hybrid, input_dim must match conv_kernel_size**2.")
        self.conv = ConvFilter(kernel_size=conv_kernel_size, threshold=threshold)
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.kernel = Kernel(gamma)
        self.fc = nn.Linear(hidden_dim + 1, hidden_dim)
        self.output = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, seq_len, input_dim).  Input_dim must be 4 for a 2×2 patch.
        Returns
        -------
        torch.Tensor
            Softmax probabilities over two classes, shape (batch, 2).
        """
        batch, seq_len, _ = x.shape

        # Convolutional feature map: mean sigmoid activation for each 2×2 patch
        conv_out = torch.zeros(batch, seq_len, device=x.device)
        for b in range(batch):
            for t in range(seq_len):
                patch = x[b, t].view(2, 2).cpu().numpy()
                conv_out[b, t] = self.conv.run(patch)

        # LSTM sequence modelling
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]  # shape (batch, hidden_dim)

        # Kernel similarity against a zero reference vector
        zero_ref = torch.zeros(batch, x.shape[-1], device=x.device)
        kernel_sim = self.kernel(x.reshape(batch, -1), zero_ref).unsqueeze(-1)  # shape (batch, 1)

        # Concatenate features
        combined = torch.cat([conv_out.mean(dim=1, keepdim=True), last_hidden, kernel_sim], dim=-1)

        # Classification head
        hidden = F.relu(self.fc(combined))
        logits = self.output(hidden)
        probs = F.softmax(logits, dim=-1)
        return probs

__all__ = ["HybridSamplerQNN"]
