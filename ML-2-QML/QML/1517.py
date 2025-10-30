import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
from typing import Tuple, Optional

class QGate(tq.QuantumModule):
    """
    A lightweight variational quantum gate that maps a classical
    input vector to a probability distribution over qubit outcomes.
    The circuit consists of parameterised rotations followed by an
    optional entanglement depth and measurement in a chosen basis.
    """

    def __init__(self,
                 n_wires: int,
                 basis: str = 'z',
                 entanglement_depth: int = 1) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.basis = basis
        self.entanglement_depth = entanglement_depth

        # Parameterised rotation gates (one per wire)
        self.rotation = nn.ModuleList([tq.RX(has_params=True) for _ in range(n_wires)])

        if basis == 'z':
            self.measure = tq.MeasureAll(tq.PauliZ)
        elif basis == 'x':
            self.measure = tq.MeasureAll(tq.PauliX)
        else:
            raise ValueError(f"Unsupported basis {basis}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape (batch, n_wires) containing rotation angles.

        Returns
        -------
        probs : torch.Tensor
            Probabilities in [0, 1] for each qubit, shape (batch, n_wires).
        """
        batch_size = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(n_wires=self.n_wires,
                                bsz=batch_size,
                                device=device)

        # Apply rotations
        for idx, gate in enumerate(self.rotation):
            gate(qdev, wires=idx, angle=x[:, idx])

        # Entanglement
        for _ in range(self.entanglement_depth):
            for i in range(self.n_wires - 1):
                tqf.cnot(qdev, wires=[i, i + 1])
            # close the ring
            tqf.cnot(qdev, wires=[self.n_wires - 1, 0])

        # Measurement
        meas = self.measure(qdev)          # shape (batch, n_wires) with values -1/1
        probs = (meas + 1) / 2.0           # map to [0, 1]
        return probs


class QLSTM(nn.Module):
    """
    Hybrid quantum‑classical LSTM cell where each gate is a variational
    quantum circuit followed by a linear mapping to the hidden dimension.
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 n_qubits: int,
                 entanglement_depth: int = 1,
                 basis: str = 'z') -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.entanglement_depth = entanglement_depth
        self.basis = basis

        # Quantum gates for the four LSTM gates
        self.forget_gate = QGate(n_qubits, basis, entanglement_depth)
        self.input_gate = QGate(n_qubits, basis, entanglement_depth)
        self.update_gate = QGate(n_qubits, basis, entanglement_depth)
        self.output_gate = QGate(n_qubits, basis, entanglement_depth)

        # Linear projections from concatenated input+hidden to quantum register
        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

        # Map quantum output to hidden dimension
        self.gate_to_hidden = nn.Linear(n_qubits, hidden_dim)

    def forward(self,
                inputs: torch.Tensor,
                states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Parameters
        ----------
        inputs : torch.Tensor
            Sequence of shape (seq_len, batch, input_dim).
        states : tuple of torch.Tensor or None
            Initial hidden and cell state.  If None, zeros are used.

        Returns
        -------
        outputs : torch.Tensor
            Hidden states for each time step (seq_len, batch, hidden_dim).
        (hx, cx) : tuple
            Final hidden and cell states.
        """
        hx, cx = self._init_states(inputs, states)
        outputs = []

        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)

            f_raw = self.forget_gate(self.linear_forget(combined))
            i_raw = self.input_gate(self.linear_input(combined))
            g_raw = self.update_gate(self.linear_update(combined))
            o_raw = self.output_gate(self.linear_output(combined))

            f = torch.sigmoid(self.gate_to_hidden(f_raw))
            i = torch.sigmoid(self.gate_to_hidden(i_raw))
            g = torch.tanh(self.gate_to_hidden(g_raw))
            o = torch.sigmoid(self.gate_to_hidden(o_raw))

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))

        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(self,
                     inputs: torch.Tensor,
                     states: Optional[Tuple[torch.Tensor, torch.Tensor]]
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx


class LSTMTagger(nn.Module):
    """
    Sequence tagging model that can switch between the classical
    :class:`QLSTM` and the quantum‑enhanced :class:`QLSTM`.
    The interface is identical to the original seed.
    """

    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 n_qubits: int = 0,
                 entanglement_depth: int = 1,
                 basis: str = 'z') -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim,
                              hidden_dim,
                              n_qubits=n_qubits,
                              entanglement_depth=entanglement_depth,
                              basis=basis)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a single sentence.

        Parameters
        ----------
        sentence : torch.Tensor
            Tensor of word indices of shape (seq_len, batch).

        Returns
        -------
        log_probs : torch.Tensor
            Log‑probabilities for each tag at each position.
        """
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return torch.nn.functional.log_softmax(tag_logits, dim=1)


__all__ = ['QLSTM', 'LSTMTagger']
