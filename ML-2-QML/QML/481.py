"""Quantum feed-forward module for hybrid transformer.

The module implements a parameterised quantum circuit that receives a
token embedding of dimensionality `n_qubits`.  Each element of the
embedding is encoded into a rotation on a dedicated qubit.  After
applying a trainable RX circuit, the qubits are measured in the Z
basis.  The resulting expectation values are passed through two
classical linear layers to form a feed‑forward transformation that
matches the hidden dimension of the transformer.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class QuantumFeedForward(tq.QuantumModule):
    """Parameterised quantum feed‑forward network.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (and therefore the dimensionality of the
        input token embedding that will be encoded).
    ffn_dim : int
        Hidden dimension of the classical linear layers that follow
        the quantum measurement.
    dropout : float, default 0.1
        Dropout probability applied after the first linear layer.
    q_device : Optional[tq.QuantumDevice], default None
        Quantum device used for the simulation.  If ``None`` a new
        device is created for every forward pass.
    """
    def __init__(self,
                 n_qubits: int,
                 ffn_dim: int,
                 dropout: float = 0.1,
                 q_device: tq.QuantumDevice | None = None) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

        # Encode each input dimension into a rotation on a dedicated qubit.
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
        )
        # Trainable RX gates applied after encoding
        self.parameters = nn.ModuleList([tq.RX(has_params=True) for _ in range(n_qubits)])
        # Measure all qubits in the Z basis
        self.measure = tq.MeasureAll(tq.PauliZ)

        # Classical layers mapping quantum output to transformer hidden space
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, n_qubits)

        # Optional external quantum device
        self.q_device = q_device

    def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice | None = None) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq, n_qubits).
        q_device : Optional[tq.QuantumDevice]
            Quantum device to use for simulation.  If ``None`` a new
            device with ``n_wires = self.n_qubits`` is created.
        """
        # Create a quantum device if one is not supplied
        if q_device is None:
            q_device = self.q_device or tq.QuantumDevice(n_wires=self.n_qubits)
        else:
            # Ensure the supplied device has the correct number of wires
            if q_device.n_wires!= self.n_qubits:
                raise ValueError(
                    f"Provided quantum device has {q_device.n_wires} wires, "
                    f"but {self.n_qubits} wires are required."
                )

        # Encode classical information into the quantum state
        self.encoder(q_device, x)
        # Apply trainable circuit
        for wire, gate in enumerate(self.parameters):
            gate(q_device, wires=wire)
        # Measure all qubits
        out = self.measure(q_device)  # shape (batch, seq, n_qubits)
        # Classical mapping
        out = self.linear1(self.dropout(out))
        out = self.linear2(F.relu(out))
        return out

__all__ = ["QuantumFeedForward"]
