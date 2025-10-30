import torch
import torch.nn as nn
import torch.nn.functional as F

class _QLayerSim(nn.Module):
    """A lightweight classical surrogate for a quantum gate.
    It applies a linear transformation followed by a tanh non‑linearity
    to emulate the effect of a measurement on a parameterised quantum
    circuit while remaining fully differentiable."""
    def __init__(self, n_qubits: int):
        super().__init__()
        self.linear = nn.Linear(n_qubits, n_qubits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.linear(x))

class QLSTMHybrid(nn.Module):
    """Hybrid LSTM that can toggle between a pure classical implementation
    and a quantum‑augmented variant via the ``n_qubits`` flag.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input at each time step.
    hidden_dim : int
        Size of the LSTM hidden state.
    n_qubits : int, default 0
        When >0 the gates are processed through a simulated quantum layer
        before the standard linear transformations.  Setting ``n_qubits=0``
        yields a conventional LSTM.
    dropout : float, default 0.0
        Dropout probability applied to the hidden state after each update.
    use_layernorm : bool, default False
        Whether to apply LayerNorm to each gate before the cell update.
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0,
                 dropout: float = 0.0, use_layernorm: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.dropout = nn.Dropout(dropout)
        self.use_layernorm = use_layernorm

        # Classical linear gates
        self.forget_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)

        if n_qubits > 0:
            self.forget_q = _QLayerSim(n_qubits)
            self.input_q = _QLayerSim(n_qubits)
            self.update_q = _QLayerSim(n_qubits)
            self.output_q = _QLayerSim(n_qubits)
            self.quantum_to_hidden = nn.Linear(n_qubits, hidden_dim)
        else:
            self.forget_q = self.input_q = self.update_q = self.output_q = None

        if use_layernorm:
            self.ln_f = nn.LayerNorm(hidden_dim)
            self.ln_i = nn.LayerNorm(hidden_dim)
            self.ln_u = nn.LayerNorm(hidden_dim)
            self.ln_o = nn.LayerNorm(hidden_dim)

    def _init_states(self, inputs: torch.Tensor,
                     states: tuple[torch.Tensor, torch.Tensor] | None = None):
        """Initialise hidden and cell states."""
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

    def forward(self, inputs: torch.Tensor,
                states: tuple[torch.Tensor, torch.Tensor] | None = None
                ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Run the LSTM over a sequence.

        Parameters
        ----------
        inputs : torch.Tensor
            Sequence of shape ``(seq_len, batch, input_dim)``.
        states : tuple, optional
            Pre‑initialised hidden and cell states.

        Returns
        -------
        output : torch.Tensor
            Hidden states over time ``(seq_len, batch, hidden_dim)``.
        (hx, cx) : tuple
            Final hidden and cell states.
        """
        hx, cx = self._init_states(inputs, states)
        outputs = []

        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)

            f = torch.sigmoid(self.forget_linear(combined))
            i = torch.sigmoid(self.input_linear(combined))
            g = torch.tanh(self.update_linear(combined))
            o = torch.sigmoid(self.output_linear(combined))

            if self.n_qubits > 0:
                f_q = self.forget_q(f)
                i_q = self.input_q(i)
                g_q = self.update_q(g)
                o_q = self.output_q(o)

                f = torch.sigmoid(self.quantum_to_hidden(f_q))
                i = torch.sigmoid(self.quantum_to_hidden(i_q))
                g = torch.tanh(self.quantum_to_hidden(g_q))
                o = torch.sigmoid(self.quantum_to_hidden(o_q))

            if self.use_layernorm:
                f = self.ln_f(f)
                i = self.ln_i(i)
                g = self.ln_u(g)
                o = self.ln_o(o)

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            hx = self.dropout(hx)

            outputs.append(hx.unsqueeze(0))

        output = torch.cat(outputs, dim=0)
        return output, (hx, cx)

__all__ = ['QLSTMHybrid']
