import numpy as np
import torch
from torch import nn

class FullyConnectedLayer(nn.Module):
    """
    Classical fully‑connected layer with optional hidden depths.
    Parameters
    ----------
    n_features : int
        Number of input features.
    hidden_sizes : list[int] | None
        Sizes of hidden layers. If None, a single linear layer is used.
    activation : nn.Module | None
        Activation function between layers. Defaults to nn.Tanh.
    """
    def __init__(self, n_features: int = 1, hidden_sizes: list[int] | None = None,
                 activation: nn.Module | None = nn.Tanh()):
        super().__init__()
        layers = []
        in_features = n_features
        if hidden_sizes:
            for h in hidden_sizes:
                layers.append(nn.Linear(in_features, h))
                if activation is not None:
                    layers.append(activation)
                in_features = h
        layers.append(nn.Linear(in_features, 1))
        self.net = nn.Sequential(*layers)

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """
        Forward pass using externally provided parameters.
        Thetas are expected to be a flat array matching all network weights
        and biases in the order they appear in the Sequential module.
        """
        # Flatten all parameters
        params = []
        for param in self.parameters():
            params.append(param.data.view(-1))
        param_flat = torch.cat(params)
        # sanity check
        if param_flat.numel()!= thetas.size:
            raise ValueError(f"Expected {param_flat.numel()} parameters, got {thetas.size}.")
        # Assign new values
        idx = 0
        for param in self.parameters():
            numel = param.numel()
            new_vals = torch.from_numpy(thetas[idx:idx+numel]).reshape(param.shape)
            param.data.copy_(new_vals)
            idx += numel
        # Forward
        with torch.no_grad():
            out = self.net(torch.as_tensor(thetas, dtype=torch.float32).unsqueeze(0))
        return out.squeeze().detach().numpy()

    def gradient(self, thetas: np.ndarray) -> np.ndarray:
        """
        Compute gradient of the output w.r.t. the parameters using autograd.
        """
        # Reset parameters
        idx = 0
        for param in self.parameters():
            numel = param.numel()
            new_vals = torch.from_numpy(thetas[idx:idx+numel]).reshape(param.shape)
            param.data.copy_(new_vals)
            idx += numel

        # Enable gradient tracking
        for param in self.parameters():
            param.requires_grad_(True)
        x = torch.as_tensor(thetas, dtype=torch.float32).unsqueeze(0)
        out = self.net(x)
        out.backward()
        grads = [p.grad.detach().numpy().reshape(-1) for p in self.parameters()]
        return np.concatenate(grads)

__all__ = ["FullyConnectedLayer"]
