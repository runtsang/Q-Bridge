import numpy as np
import torch
from torch import nn

class HybridClassifier(nn.Module):
    """Hybrid classical model that mimics a fully‑connected layer with optional depth.

    The network is built from a stack of linear layers followed by ReLU
    activations, ending with a single‑output linear layer.  Parameters
    are supplied externally via ``run(thetas)`` so that the same
    interface can be used for the quantum counterpart.  The design
    follows the classical seed from *FCL.py* and the depth‑controlled
    architecture from *QuantumClassifierModel.py*.
    """
    def __init__(self, n_features: int = 1, depth: int = 1):
        super().__init__()
        self.n_features = n_features
        self.depth = depth
        self._build_network()

    def _build_network(self):
        layers = []
        in_dim = self.n_features
        for _ in range(self.depth):
            linear = nn.Linear(in_dim, self.n_features)
            layers.extend([linear, nn.ReLU()])
        # final output layer
        layers.append(nn.Linear(self.n_features, 1))
        self.network = nn.Sequential(*layers)

        # flatten weight sizes to match the order expected by ``run``
        self.weight_sizes = []
        for m in self.network:
            if isinstance(m, nn.Linear):
                self.weight_sizes.append(m.weight.numel() + m.bias.numel())

    def _set_parameters(self, thetas: np.ndarray):
        """Map a flat list of parameters onto the network weights."""
        idx = 0
        for m in self.network:
            if isinstance(m, nn.Linear):
                w_size = m.weight.numel()
                b_size = m.bias.numel()
                m.weight.data = torch.from_numpy(
                    thetas[idx:idx + w_size].reshape(m.weight.shape)
                ).float()
                idx += w_size
                m.bias.data = torch.from_numpy(
                    thetas[idx:idx + b_size].reshape(m.bias.shape)
                ).float()
                idx += b_size
        if idx!= len(thetas):
            raise ValueError("Parameter vector length does not match network size")

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """Execute the network with the supplied parameters.

        Parameters
        ----------
        thetas : np.ndarray
            Flat array of shape ``sum(weight_sizes)`` containing all weights
            and biases.

        Returns
        -------
        np.ndarray
            1‑D array containing the scalar output of the network.
        """
        thetas = np.asarray(thetas, dtype=np.float32).flatten()
        self._set_parameters(thetas)
        # deterministic input – zeros – to keep the output independent of data
        dummy = torch.zeros(1, self.n_features, dtype=torch.float32)
        out = self.network(dummy).detach().numpy().reshape(-1)
        return out

def FCL(n_features: int = 1, depth: int = 1, mode: str = "classical") -> HybridClassifier:
    """Factory returning a hybrid classifier in classical mode.

    Parameters
    ----------
    n_features : int
        Dimensionality of the input feature space.
    depth : int
        Number of hidden layers.
    mode : str
        Currently only ``"classical"`` is supported; the quantum variant is
        provided in the QML module.

    Returns
    -------
    HybridClassifier
        Instance configured for classical execution.
    """
    if mode!= "classical":
        raise ValueError("Only classical mode is supported in the ML module.")
    return HybridClassifier(n_features, depth)

__all__ = ["HybridClassifier", "FCL"]
