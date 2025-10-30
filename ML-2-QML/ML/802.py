"""Enhanced classical fully‑connected layer with dropout and batch‑norm.

The module exposes a single callable ``FCL`` that returns a PyTorch
``nn.Module``.  The ``run`` method accepts a list of floats that are
directly mapped to the linear layer’s weight matrix (and bias if the
list is long enough).  This makes it trivial to use the class as a
drop‑in replacement for the original quantum‑style interface while
providing full back‑prop support for downstream training pipelines.

Typical usage::

    fcl = FCL(n_features=10, n_hidden=20)
    out = fcl.run([0.1]*10)  # overwrite weights with 0.1

"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
from torch import nn


class FullyConnectedLayer(nn.Module):
    """
    A classical feed‑forward network that mimics the API of the quantum
    reference.  It consists of a single linear layer followed by
    dropout and batch‑norm.  The ``run`` method accepts an iterable of
    parameters that are reshaped to match the weight matrix.
    """

    def __init__(
        self,
        n_features: int = 1,
        n_hidden: int | None = None,
        dropout_rate: float = 0.0,
        use_batchnorm: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        n_features:
            Dimensionality of the input.
        n_hidden:
            Optional hidden size; if ``None`` a single linear layer is used.
        dropout_rate:
            Dropout probability applied after the linear layer.
        use_batchnorm:
            Whether to add a BatchNorm1d layer.
        """
        super().__init__()
        layers: list[nn.Module] = []

        if n_hidden is None:
            layers.append(nn.Linear(n_features, 1))
        else:
            layers.append(nn.Linear(n_features, n_hidden))
            layers.append(nn.ReLU())
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(n_hidden))
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))
            layers.append(nn.Linear(n_hidden, 1))

        self.model = nn.Sequential(*layers)

    def run(self, thetas: Iterable[float]) -> torch.Tensor:
        """
        Forward pass that replaces the linear layer weights with the
        supplied parameters.  The method returns a 1‑D tensor of the
        output values.

        Parameters
        ----------
        thetas:
            Iterable of floats that will be reshaped to match the
            weight matrix of the first linear layer.  If the iterable
            contains more values than needed, the surplus are ignored.
            If fewer values are supplied, the remaining weights are
            left unchanged.

        Returns
        -------
        torch.Tensor
            The network output after applying ``tanh`` and taking the
            mean across the batch dimension.
        """
        # Extract the first linear layer
        linear = next(
            (m for m in self.model.modules() if isinstance(m, nn.Linear)), None
        )
        if linear is None:
            raise RuntimeError("No linear layer found in the model.")

        # Flatten current weights for easy replacement
        weight_shape = linear.weight.shape
        bias_shape = linear.bias.shape if linear.bias is not None else None

        # Flatten the provided parameters
        params = torch.as_tensor(list(thetas), dtype=torch.float32)

        # Replace weights
        expected_weight_size = weight_shape.numel()
        if params.numel() >= expected_weight_size:
            linear.weight.data = params[:expected_weight_size].view(weight_shape)
        else:
            # Pad with existing weights if insufficient
            linear.weight.data = torch.cat(
                (params.view(-1, 1), linear.weight.data[expected_weight_size:]),
                dim=0,
            ).view(weight_shape)

        # Replace bias if provided
        if bias_shape is not None and params.numel() > expected_weight_size:
            bias_params = params[expected_weight_size : expected_weight_size + bias_shape.numel()]
            linear.bias.data = bias_params.view(bias_shape)

        # Forward pass
        output = self.model(torch.randn(1, self.model[0].in_features))
        expectation = torch.tanh(output).mean(dim=0)
        return expectation.detach()


def FCL(
    n_features: int = 1,
    n_hidden: int | None = None,
    dropout_rate: float = 0.0,
    use_batchnorm: bool = False,
) -> FullyConnectedLayer:
    """
    Factory function that returns an instance of ``FullyConnectedLayer``.
    The signature mirrors the original seed, but additional optional
    arguments allow for richer experimentation.

    Returns
    -------
    FullyConnectedLayer
    """
    return FullyConnectedLayer(
        n_features=n_features,
        n_hidden=n_hidden,
        dropout_rate=dropout_rate,
        use_batchnorm=use_batchnorm,
    )


__all__ = ["FCL"]
