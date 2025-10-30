import torch
import torch.nn as nn
import numpy as np

class HybridFCL(nn.Module):
    """
    Classical fully‑connected layer with multiple linear stages.
    The network accepts a flat parameter vector that is reshaped into
    the weight matrices and biases of each linear layer.
    Dropout and batch‑normalisation are added to increase robustness.
    """

    def __init__(self, layer_sizes: list[int] = [1, 1]):
        """
        Parameters
        ----------
        layer_sizes : list[int]
            Size of each layer including input and output.
            Example: [1, 8, 1] creates a 1→8→1 network.
        """
        super().__init__()
        self.layers = nn.ModuleList()
        for in_f, out_f in zip(layer_sizes[:-1], layer_sizes[1:]):
            self.layers.append(nn.Linear(in_f, out_f))
        self.dropout = nn.Dropout(p=0.2)
        self.bn = nn.BatchNorm1d(layer_sizes[-1])

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """
        Apply the network to the supplied parameters and return a scalar.
        The flat vector `thetas` must match the total number of trainable
        parameters in the network. The method reshapes the vector into
        individual weight matrices and bias vectors, updates the layers,
        and performs a forward pass.
        """
        params = torch.from_numpy(thetas.astype(np.float32))
        idx = 0
        x = torch.tensor([params[0]], dtype=torch.float32).view(1, -1)
        for layer in self.layers:
            w_size = layer.weight.numel()
            b_size = layer.bias.numel()
            w = params[idx:idx + w_size].view(layer.in_features, layer.out_features)
            idx += w_size
            b = params[idx:idx + b_size]
            idx += b_size
            layer.weight.data = w.clone()
            layer.bias.data = b.clone()
            x = layer(x)
            x = torch.tanh(x)
        x = self.dropout(x)
        x = self.bn(x)
        return x.detach().cpu().numpy().flatten()
