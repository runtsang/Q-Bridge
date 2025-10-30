import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Tuple, List

class QuantumClassifierModel:
    """
    Classical feed‑forward classifier with optional residual connections and
    configurable activation functions.  The class exposes a ``build_classifier_circuit``
    factory that mirrors the quantum helper interface, returning the network,
    a list of input indices, a list of weight counts, and the output indices.
    """

    def __init__(self,
                 num_features: int,
                 depth: int = 4,
                 activation: str = "relu",
                 residual: bool = True):
        self.num_features = num_features
        self.depth = depth
        self.activation = activation
        self.residual = residual
        self.network, self.encoding, self.weight_sizes, self.observables = self.build_classifier_circuit()

    @staticmethod
    def _activation_fn(name: str):
        return getattr(F, name) if hasattr(F, name) else nn.Identity()

    @classmethod
    def build_classifier_circuit(cls,
                                 num_features: int,
                                 depth: int = 4,
                                 activation: str = "relu",
                                 residual: bool = True,
                                 **kwargs) -> Tuple[nn.Module,
                                                    Iterable[int],
                                                    Iterable[int],
                                                    List[int]]:
        """
        Construct a feed‑forward network and return metadata.
        """
        layers: List[nn.Module] = []
        in_dim = num_features
        encoding: List[int] = list(range(num_features))
        weight_sizes: List[int] = []

        act_fn = cls._activation_fn(activation)

        for _ in range(depth):
            linear = nn.Linear(in_dim, num_features)
            if residual and in_dim == num_features:
                # Residual block: Linear -> Act -> Linear
                res_block = nn.Sequential(
                    linear,
                    act_fn(),
                    nn.Linear(num_features, num_features)
                )
                layers.append(res_block)
                weight_sizes.append(
                    linear.weight.numel() + linear.bias.numel() +
                    res_block[2].weight.numel() + res_block[2].bias.numel()
                )
            else:
                layers.append(linear)
                layers.append(act_fn())
                weight_sizes.append(linear.weight.numel() + linear.bias.numel())
            in_dim = num_features

        head = nn.Linear(in_dim, 2)
        layers.append(head)
        weight_sizes.append(head.weight.numel() + head.bias.numel())

        network = nn.Sequential(*layers)
        observables: List[int] = [0, 1]  # output class indices
        return network, encoding, weight_sizes, observables

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
