"""Fully connected layer with optional quantum augmentation.

This module defines an `FCL` class that acts as a classical dense layer
and can optionally delegate its forward pass to a quantum circuit.
The quantum part is completely decoupled: the user passes a ready‑to‑run
quantum circuit that accepts a list of parameters and returns a numpy
array.  The class then converts the output back to a torch tensor so it
can be used in a normal neural network.
"""

import torch
from torch import nn
import numpy as np

class FCL(nn.Module):
    """Modular fully‑connected layer with optional quantum support.

    Parameters
    ----------
    n_features : int
        Number of input features.
    activation : str, optional
        Activation function to apply after the linear transform.
        Supported values are ``'relu'``, ``'tanh'``, ``'sigmoid'`` and
        ``'none'`` (identity).  Default is ``'relu'``.
    use_quantum : bool, optional
        If ``True`` the output of the linear layer is fed to a quantum
        circuit supplied via ``quantum_circuit``.  The returned value
        is the circuit's expectation value.
    quantum_circuit : object, optional
        An object exposing a ``run(list[float]) -> np.ndarray`` method.
        The list of parameters fed to the circuit is taken from the
        linear output.  The shape of the returned array is assumed to
        be ``(1,)``.
    dropout : float, optional
        Dropout probability applied to the layer's output.  Set to ``0``
        for no dropout.
    """

    def __init__(
        self,
        n_features: int,
        activation: str = "relu",
        use_quantum: bool = False,
        quantum_circuit=None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)
        act_dict = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "none": nn.Identity(),
        }
        self.activation = act_dict.get(activation, nn.ReLU())
        self.use_quantum = use_quantum
        self.quantum_circuit = quantum_circuit
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass.

        If ``use_quantum`` is ``True`` and a quantum circuit is
        provided, the linear output is interpreted as a list of
        parameters for the circuit.  The circuit's expectation value
        is returned as a 1‑D tensor.
        """
        out = self.linear(x)
        out = self.activation(out)
        out = self.dropout(out)

        if self.use_quantum and self.quantum_circuit is not None:
            # Convert the linear output to a list of parameters.
            params = out.detach().cpu().numpy().flatten().tolist()
            q_out = self.quantum_circuit.run(params)
            return torch.tensor(q_out, dtype=x.dtype, device=x.device)

        return out

    def set_quantum_circuit(self, qc):
        """Set or replace the quantum circuit used in the forward pass."""
        self.quantum_circuit = qc

__all__ = ["FCL"]
