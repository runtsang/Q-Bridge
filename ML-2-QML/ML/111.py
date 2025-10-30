import torch
import torch.nn as nn
import torch.nn.functional as F

class QLSTM(nn.Module):
    """
    Classical LSTM cell with optional quantum‑augmented gates.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input at each time step.
    hidden_dim : int
        Dimensionality of the hidden state.
    n_qubits : int, default 0
        Number of qubits used if a quantum gate is enabled.
    q_depth : int, default 0
        Depth of the quantum circuit per gate.  Zero disables the quantum component.
    reg_weight : float, default 1e-3
        Weight of the regulariser that penalises large quantum outputs.
    """
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 n_qubits: int = 0,
                 q_depth: int = 0,
                 reg_weight: float = 1e-3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.q_depth = q_depth
        self.reg_weight = reg_weight

        # Classical linear maps for all gates
        self.forget_lin = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_lin  = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_lin = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_lin = nn.Linear(input_dim + hidden_dim, hidden_dim)

        # Optional quantum modules per gate
        if self.n_qubits > 0 and self.q_depth > 0:
            self.forget_q = self._make_qgate()
            self.input_q  = self._make_qgate()
            self.update_q  = self._make_qgate()
            self.output_q  = self._make_qgate()
        else:
            self.forget_q = self.input_q = self.update_q = self.output_q = None

    def _make_qgate(self):
        """
        Builds a shallow quantum circuit that returns a tensor of size (batch, hidden_dim).
        The circuit is parameter‑shared across all time steps and gates.
        """
        import torchquantum as tq
        import torchquantum.functional as tqf
        from torch.nn.parameter import Parameter

        # Parameter matrix of shape (hidden_dim, n_qubits)
        param = Parameter(torch.randn(self.hidden_dim, self.n_qubits))

        def gate(x: torch.Tensor):
            # x shape: (batch, hidden_dim)
            batch_size = x.shape[0]
            device = x.device
            qdev = tq.QuantumDevice(n_wires=self.n_qubits,
                                    bsz=batch_size,
                                    device=device)

            # Encode each hidden unit into one qubit (first n_qubits)
            for i in range(min(self.hidden_dim, self.n_qubits)):
                tqf.rx(qdev, wires=[i], params=x[:, i])

            # Apply depth‑controlled parameterised layers
            for _ in range(self.q_depth):
                for i in range(self.n_qubits):
                    tqf.rz(qdev, wires=[i], params=param[:, i % self.n_qubits])
                    tqf.cnot(qdev, wires=[i, (i + 1) % self.n_qubits])

            # Measure all qubits in Z basis
            meas = tq.MeasureAll(tq.PauliZ)(qdev)
            # Convert measurement (0/1) to [-1,1] range
            return 2 * meas - 1

        return gate

    def forward(self,
                inputs: torch.Tensor,
                states: tuple[torch.Tensor, torch.Tensor] | None = None):
        """
        Parameters
        ----------
        inputs : torch.Tensor
            Input sequence of shape (seq_len, batch, input_dim).
        states : tuple, optional
            Tuple of (hx, cx) hidden and cell states.

        Returns
        -------
        outputs : torch.Tensor
            Sequence of hidden states (seq_len, batch, hidden_dim).
        state : tuple
            Final hidden and cell states.
        """
        hx, cx = self._init_states(inputs, states)
        outputs = []

        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)

            f = torch.sigmoid(self.forget_lin(combined))
            i = torch.sigmoid(self.input_lin(combined))
            g = torch.tanh(self.update_lin(combined))
            o = torch.sigmoid(self.output_lin(combined))

            # Add quantum augmentation if enabled
            if self.forget_q is not None:
                f = f * torch.sigmoid(self.forget_q(f))
                i = i * torch.sigmoid(self.input_q(i))
                g = g * torch.tanh(self.update_q(g))
                o = o * torch.sigmoid(self.output_q(o))

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)

            # Regularisation term on quantum outputs
            if self.forget_q is not None:
                reg = (f + i + g + o).abs().mean()
                hx = hx - self.reg_weight * reg

            outputs.append(hx.unsqueeze(0))

        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(self,
                     inputs: torch.Tensor,
                     states: tuple[torch.Tensor, torch.Tensor] | None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))

class LSTMTagger(nn.Module):
    """
    Sequence tagging model that can switch between the enhanced classical
    LSTM and a standard PyTorch LSTM.  The interface is identical to the
    original implementation, so downstream code can swap modules without
    modification.
    """
    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 n_qubits: int = 0,
                 q_depth: int = 0,
                 reg_weight: float = 1e-3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        if n_qubits > 0 and q_depth > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim,
                              n_qubits=n_qubits,
                              q_depth=q_depth,
                              reg_weight=reg_weight)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor):
        """
        Parameters
        ----------
        sentence : torch.Tensor
            Tensor of word indices, shape (seq_len, batch).

        Returns
        -------
        log_probs : torch.Tensor
            Log‑probabilities over tags for each token.
        """
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["QLSTM", "LSTMTagger"]
