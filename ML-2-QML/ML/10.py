import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class QLSTMGen(nn.Module):
    """
    Classical LSTM cell that optionally replaces each gate with a
    lightweight variational quantum circuit.  The quantum part is
    simulated with simple trigonometric transformations to keep the
    module fully PyTorch‑based while preserving the interface of the
    original QLSTM.  The class supports freezing the quantum
    parameters for classical fine‑tuning or joint training.
    """
    class _QuantumGate(nn.Module):
        """
        Simulated quantum gate: a single‑qubit rotation followed by a
        parameterized amplitude function.  The gate maps an input
        vector of size ``n_qubits`` to a probability vector of the
        same length.
        """
        def __init__(self, n_qubits: int):
            super().__init__()
            # Each qubit has a rotation angle and a scaling factor
            self.theta = nn.Parameter(torch.randn(n_qubits))
            self.scale = nn.Parameter(torch.rand(n_qubits))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Parameters
            ----------
            x : torch.Tensor
                Input tensor of shape (batch, n_qubits).
            Returns
            -------
            torch.Tensor
                Activated output of shape (batch, n_qubits).
            """
            # Simulate rotation around X and map to [0,1] by sigmoid
            rot = torch.sin(x * self.theta) * self.scale
            return torch.sigmoid(rot)

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int,
        freeze_quantum: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        input_dim : int
            Dimensionality of the input embeddings.
        hidden_dim : int
            Dimensionality of the hidden state.
        n_qubits : int
            Number of qubits (and thus output dimensions) for each gate.
        freeze_quantum : bool, optional
            If True, quantum gate parameters are frozen during training.
            Default is False.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.freeze_quantum = freeze_quantum

        # Linear projections for each gate
        self.forget_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.input_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.update_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.output_lin = nn.Linear(input_dim + hidden_dim, n_qubits)

        # Quantum gates
        self.forget_gate = self._QuantumGate(n_qubits)
        self.input_gate = self._QuantumGate(n_qubits)
        self.update_gate = self._QuantumGate(n_qubits)
        self.output_gate = self._QuantumGate(n_qubits)

        # Optionally freeze quantum parameters
        if self.freeze_quantum:
            for p in self.parameters():
                p.requires_grad = False
            # but keep the linear layers trainable
            for p in self.forget_lin.parameters():
                p.requires_grad = True
            for p in self.input_lin.parameters():
                p.requires_grad = True
            for p in self.update_lin.parameters():
                p.requires_grad = True
            for p in self.output_lin.parameters():
                p.requires_grad = True

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of the hybrid LSTM.

        Parameters
        ----------
        inputs : torch.Tensor
            Input sequence of shape (seq_len, batch, input_dim).
        states : tuple of torch.Tensor, optional
            Tuple (h_0, c_0) of initial hidden and cell states.
            Each of shape (batch, hidden_dim).  If None, zeros are used.

        Returns
        -------
        tuple
            - outputs : torch.Tensor of shape (seq_len, batch, hidden_dim)
            - (h_n, c_n) : final hidden and cell states.
        """
        h, c = self._init_states(inputs, states)
        outputs = []

        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, h], dim=1)  # (batch, input+hidden)
            f = torch.sigmoid(self.forget_gate(self.forget_lin(combined)))
            i = torch.sigmoid(self.input_gate(self.input_lin(combined)))
            g = torch.tanh(self.update_gate(self.update_lin(combined)))
            o = torch.sigmoid(self.output_gate(self.output_lin(combined)))

            c = f * c + i * g
            h = o * torch.tanh(c)
            outputs.append(h.unsqueeze(0))

        outputs = torch.cat(outputs, dim=0)
        return outputs, (h, c)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

__all__ = ["QLSTMGen"]
