"""
Hybrid sampler/estimator network that combines classical and quantum
components.  The architecture is a direct synthesis of the four
reference seeds:

* SamplerQNN – soft‑max output for categorical sampling.
* EstimatorQNN – regression head for continuous targets.
* FCL – single‑qubit fully‑connected layer that can be swapped for a
  quantum circuit.
* QuantumRegression – quantum module that extracts features from
  encoded states.

The class ``SamplerQNNGen312`` is fully torch‑compatible and can be
used in any standard training loop.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# The quantum module lives in the companion QML file and is imported
# lazily to avoid heavy dependencies when the user only needs the
# classical part.
try:
    from.quantum_sampler import QuantumSampler
except Exception:  # pragma: no cover
    QuantumSampler = None  # type: ignore[assignment]


class SamplerQNNGen312(nn.Module):
    """
    Hybrid sampler/estimator network.

    Parameters
    ----------
    n_features : int, default 2
        Number of classical input features.
    hidden_sizes : tuple[int,...], default (8, 4)
        Sizes of intermediate hidden layers in the classical backbone.
    use_quantum : bool, default True
        If ``True`` prepend a quantum feature extractor and append a
        linear head that consumes the quantum probability vector.
    n_qubits : int, default 2
        Number of qubits used by the quantum encoder.
    device : str | torch.device, default "cpu"
        Target device for tensors.
    """

    def __init__(
        self,
        n_features: int = 2,
        hidden_sizes: tuple[int,...] = (8, 4),
        use_quantum: bool = True,
        n_qubits: int = 2,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.use_quantum = use_quantum
        self.device = device

        # Classical backbone – a small MLP with Tanh activations.
        layers = []
        in_dim = n_features
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Tanh())
            in_dim = h
        self.classical_backbone = nn.Sequential(*layers)

        if use_quantum:
            if QuantumSampler is None:
                raise ImportError(
                    "QuantumSampler could not be imported. "
                    "Ensure the QML module is installed."
                )
            # Quantum sampler that maps classical inputs to a probability
            # distribution over 2^n_qubits outcomes.
            self.quantum = QuantumSampler(n_qubits=n_qubits, device=device)
            # Linear head that consumes the 2^n_qubits probability vector.
            self.head = nn.Linear(2**n_qubits, 1)
        else:
            # Purely classical estimator head.
            self.head = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, n_features).

        Returns
        -------
        torch.Tensor
            For ``use_quantum`` the output is a scalar regression value.
            Otherwise it is a soft‑max probability vector (categorical sampler).
        """
        # Classical feature extraction
        feat = self.classical_backbone(x.to(self.device))

        if self.use_quantum:
            # Encode the classical features into the quantum circuit
            probs = self.quantum(feat)  # shape (batch, 2**n_qubits)
            out = self.head(probs)
            return out.squeeze(-1)
        else:
            # Soft‑max for categorical sampling
            out = F.softmax(feat, dim=-1)
            return out
