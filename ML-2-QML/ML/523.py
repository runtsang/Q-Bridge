import torch
import torch.nn as nn
import numpy as np
from typing import Iterable, List, Callable, Optional

class HybridFCL(nn.Module):
    """
    Classical multi‑layer fully‑connected network that mimics the interface of the
    original quantum FCL example.
    Parameters
    ----------
    layer_sizes : List[int]
        Sequence of integers describing the dimensionality of each linear layer.
        The first value is the input size and the last value is the output size.
    activations : Optional[List[Callable[[torch.Tensor], torch.Tensor]]]
        Activation functions for each hidden layer.  If *None*, ``torch.tanh`` is
        used for all hidden layers.
    dropout : Optional[float]
        Drop‑out probability applied after each hidden layer.  If *None* no
        dropout is applied.
    """

    def __init__(
        self,
        layer_sizes: List[int],
        activations: Optional[List[Callable[[torch.Tensor], torch.Tensor]]] = None,
        dropout: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.activations = activations or [torch.tanh] * (len(layer_sizes) - 1)
        for inp, out, act in zip(layer_sizes[:-1], layer_sizes[1:], self.activations):
            self.layers.append(nn.Linear(inp, out))
            self.layers.append(act)
            if dropout:
                self.layers.append(nn.Dropout(dropout))

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Forward pass using a flattened list of parameters.  The parameters are
        reshaped to match the weights and biases of the underlying linear
        layers.  The method returns a NumPy array containing the network
        output.
        """
        params = torch.tensor(list(thetas), dtype=torch.float32)
        idx = 0
        # Dummy input: we only care about the parameters, so create a
        # placeholder tensor that will be overwritten by the first linear layer.
        x = torch.zeros(1, self.layers[0].in_features, dtype=torch.float32)
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                weight_shape = layer.weight.shape
                bias_shape = layer.bias.shape
                weight_size = np.prod(weight_shape)
                bias_size = np.prod(bias_shape)
                weight = params[idx : idx + weight_size].view(weight_shape)
                idx += weight_size
                bias = params[idx : idx + bias_size].view(bias_shape)
                idx += bias_size
                layer.weight = nn.Parameter(weight)
                layer.bias = nn.Parameter(bias)
            x = layer(x)
        return x.squeeze().detach().numpy()

    def train_step(self, thetas: Iterable[float], loss_fn, lr: float = 1e-3) -> float:
        """
        Perform a single gradient‑descent step on the flattened parameters.
        Returns the loss value.
        """
        params = torch.tensor(list(thetas), dtype=torch.float32, requires_grad=True)
        optimizer = torch.optim.SGD([params], lr=lr)
        optimizer.zero_grad()
        pred = self.run(params)
        loss = loss_fn(pred, torch.tensor([0.0]))
        loss.backward()
        optimizer.step()
        return loss.item()

__all__ = ["HybridFCL"]
