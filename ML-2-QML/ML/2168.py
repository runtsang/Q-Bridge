"""Enhanced classical classifier mirroring the quantum helper interface with residual and dropout layers."""
import torch
import torch.nn as nn
from typing import Iterable, Tuple, List

class QuantumClassifierModel:
    """Factory for a residual‑MLP classifier.

    The API matches the original quantum helper: a static method
    `build_classifier_circuit` returning a tuple of
    (nn.Module, encoding, weight_sizes, observables).
    The implementation now supports:
    * Residual connections between layers
    * Dropout for regularisation
    * Optional batch‑norm
    * Configurable hidden size and activation
    """
    @staticmethod
    def build_classifier_circuit(
        num_features: int,
        depth: int,
        hidden_size: int = None,
        dropout: float = 0.0,
        use_batch_norm: bool = False,
        activation: nn.Module = nn.ReLU
    ) -> Tuple[nn.Module, Iterable[int], List[int], List[int]]:
        """
        Construct a feed‑forward classifier with optional residual blocks.

        Parameters
        ----------
        num_features : int
            Size of the input feature vector.
        depth : int
            Number of residual blocks.
        hidden_size : int, optional
            Size of the hidden representations. Defaults to ``num_features``.
        dropout : float, optional
            Dropout probability applied after each block.
        use_batch_norm : bool, optional
            Whether to prepend a batch‑norm layer to each block.
        activation : nn.Module, optional
            Activation function to use inside the blocks.

        Returns
        -------
        Tuple[nn.Module, Iterable[int], List[int], List[int]]
            * ``network`` – the constructed ``nn.Sequential`` model.
            * ``encoding`` – list of feature indices used (identity mapping here).
            * ``weight_sizes`` – number of trainable parameters per block, including the final head.
            * ``observables`` – pseudo‑observables (class indices).
        """
        hidden_size = hidden_size or num_features
        weight_sizes: List[int] = []

        # initial linear mapping
        initial = nn.Linear(num_features, hidden_size)
        weight_sizes.append(initial.weight.numel() + initial.bias.numel())

        layers: List[nn.Module] = [initial]
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_size))
        layers.append(activation())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))

        in_dim = hidden_size

        # residual blocks
        for _ in range(depth):
            block = nn.Sequential(
                nn.Linear(in_dim, hidden_size),
                activation(),
                nn.Linear(hidden_size, hidden_size),
            )
            weight_sizes.append(block[0].weight.numel() + block[0].bias.numel())
            weight_sizes.append(block[2].weight.numel() + block[2].bias.numel())
            if use_batch_norm:
                block = nn.Sequential(nn.BatchNorm1d(hidden_size), block)
            layers.append(block)
            layers.append(activation())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))

        # final classifier head
        head = nn.Linear(in_dim, 2)
        layers.append(head)
        weight_sizes.append(head.weight.numel() + head.bias.numel())

        # custom module to add residuals in forward
        class ResidualClassifier(nn.Module):
            def __init__(self, layers: List[nn.Module]):
                super().__init__()
                self.layers = nn.ModuleList(layers)

            def forward(self, x):
                out = x
                idx = 0
                while idx < len(self.layers):
                    # first layer is the initial mapping
                    if idx == 0:
                        out = self.layers[idx](out)
                        idx += 1
                        continue
                    # residual block
                    block = self.layers[idx]
                    residual = out
                    out = block(out)
                    out = out + residual
                    idx += 1
                    # activation after block
                    if idx < len(self.layers):
                        out = self.layers[idx](out)
                        idx += 1
                return out

        network = ResidualClassifier(layers)

        encoding = list(range(num_features))
        observables = list(range(2))  # class indices

        return network, encoding, weight_sizes, observables

__all__ = ["QuantumClassifierModel"]
