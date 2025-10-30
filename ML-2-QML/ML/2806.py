import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple, Iterable

class UnifiedQLayer(nn.Module):
    """
    Classical dense + LSTM block that mimics the hybrid design.
    Parameters
    ----------
    input_dim : int
        Size of each time‑step input vector.
    hidden_dim : int
        Size of the hidden state that feeds the LSTM.
    n_qubits : int
        Number of qubits used in the quantum circuit (unused in classical mode).
    use_quantum_gate : bool, default False
        If True, raises ``NotImplementedError`` because quantum gates are
        defined in the QML module.
    """
    def __init__(self, input_dim: int, hidden_dim: int,
                 n_qubits: int = 0, use_quantum_gate: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.use_quantum_gate = use_quantum_gate

        if self.use_quantum_gate:
            raise NotImplementedError(
                "Quantum gates are only available in the QML module."
            )

        # Dense mapping from input to hidden dimension
        self.dense = nn.Linear(input_dim, hidden_dim)

        # Classical LSTM gates implemented with linear layers
        self.forget = nn.Linear(hidden_dim * 2, hidden_dim)
        self.input_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.update = nn.Linear(hidden_dim * 2, hidden_dim)
        self.output_gate = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, inputs: torch.Tensor,
                states: Tuple[torch.Tensor, torch.Tensor] | None = None):
        """
        Process a sequence of inputs.

        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of shape (seq_len, batch, input_dim).
        states : tuple[torch.Tensor, torch.Tensor] | None
            Optional initial hidden and cell states.
        """
        seq_len, batch, _ = inputs.shape
        hx, cx = self._init_states(batch, states)

        outputs = []
        for t in range(seq_len):
            x = inputs[t]
            # Dense mapping
            x_proj = self.dense(x)
            combined = torch.cat([x_proj, hx], dim=1)

            f = torch.sigmoid(self.forget(combined))
            i = torch.sigmoid(self.input_gate(combined))
            g = torch.tanh(self.update(combined))
            o = torch.sigmoid(self.output_gate(combined))

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))

        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(self, batch: int,
                     states: Tuple[torch.Tensor, torch.Tensor] | None):
        if states is not None:
            return states
        device = next(self.parameters()).device
        hx = torch.zeros(batch, self.hidden_dim, device=device)
        cx = torch.zeros(batch, self.hidden_dim, device=device)
        return hx, cx

__all__ = ["UnifiedQLayer"]
