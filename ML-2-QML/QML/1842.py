import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from typing import Tuple, Optional

class QLSTMGen025(nn.Module):
    """
    Quantum‑enhanced LSTM cell. Each gate is a small variational circuit
    with an input‑encoding, trainable RX layers, a chain of CNOTs,
    and a depolarising noise channel.  Supports dropout and a
    multi‑task head identical to the classical variant.
    """
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int, n_params: int = 4) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.n_params = n_params

            # Input encoding with RX gates
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )

            # Parameterised RX layers
            self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True)
                                         for _ in range(n_params)])

            # Entangling pattern: linear chain of CNOTs
            self.cnot_pattern = [(i, (i + 1) % n_wires) for i in range(n_wires)]

            # Measurement
            self.measure = tq.MeasureAll(tq.PauliZ)

            # Depolarising noise
            self.noise = tq.NoiseModule(noise_type="depolarizing",
                                         noise_rate=0.01)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            dev = tq.QuantumDevice(n_wires=self.n_wires,
                                   bsz=x.shape[0],
                                   device=x.device)

            # Encode the classical vector
            self.encoder(dev, x)

            # Apply trainable RX gates
            for gate in self.params:
                gate(dev)

            # Entangle with CNOTs
            for src, tgt in self.cnot_pattern:
                tqf.cnot(dev, wires=[src, tgt])

            # Add noise
            self.noise(dev)

            out = self.measure(dev)

            # Report gate counts to parent if available
            if hasattr(self, "_parent"):
                self._parent._update_gate_counts(self)

            return out

    def __init__(self, input_dim: int, hidden_dim: int,
                 n_qubits: int, n_tasks: int = 2,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.n_tasks = n_tasks
        self.dropout = dropout

        # Quantum gate modules
        self.forget_gate = self.QLayer(n_qubits)
        self.input_gate = self.QLayer(n_qubits)
        self.update_gate = self.QLayer(n_qubits)
        self.output_gate = self.QLayer(n_qubits)

        # Set parent reference for gate‑count tracking
        self.forget_gate._parent = self
        self.input_gate._parent = self
        self.update_gate._parent = self
        self.output_gate._parent = self

        # Linear layers to map concatenated input+hidden to qubit space
        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

        # Map quantum outputs to hidden dimension
        self.forget_map = nn.Linear(n_qubits, hidden_dim)
        self.input_map = nn.Linear(n_qubits, hidden_dim)
        self.update_map = nn.Linear(n_qubits, hidden_dim)
        self.output_map = nn.Linear(n_qubits, hidden_dim)

        # Multi‑task heads
        self.task_heads = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim)
                                         for _ in range(n_tasks)])

        self.dropout_layer = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Gate‑count statistics
        self._gate_counts = {"param_gates": 0, "entangling_gates": 0}

    def forward(self, inputs: torch.Tensor,
                seq_lengths: Optional[torch.Tensor] = None,
                states: Tuple[torch.Tensor, torch.Tensor] | None = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for a batch of sequences.

        Args:
            inputs: Tensor of shape (seq_len, batch, input_dim)
            seq_lengths: Optional tensor of shape (batch,) with actual lengths.
            states: Optional initial (hx, cx)

        Returns:
            outputs: Tensor of shape (seq_len, batch, hidden_dim)
            final_states: Tuple (hx, cx)
        """
        hx, cx = self._init_states(inputs, states)

        if seq_lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(inputs, seq_lengths.cpu(),
                                                       enforce_sorted=False)
        else:
            packed = inputs

        outputs = []
        if isinstance(packed, torch.Tensor):
            iterable = packed.unbind(dim=0)
        else:
            unpacked, _ = nn.utils.rnn.pad_packed_sequence(packed)
            iterable = unpacked.unbind(dim=0)

        for x in iterable:
            combined = torch.cat([x, hx], dim=1)

            f = torch.sigmoid(self.forget_gate(self.linear_forget(combined)))
            f = self.forget_map(f)

            i = torch.sigmoid(self.input_gate(self.linear_input(combined)))
            i = self.input_map(i)

            g = torch.tanh(self.update_gate(self.linear_update(combined)))
            g = self.update_map(g)

            o = torch.sigmoid(self.output_gate(self.linear_output(combined)))
            o = self.output_map(o)

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)

            hx = self.dropout_layer(hx)
            outputs.append(hx.unsqueeze(0))

        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(self, inputs: torch.Tensor,
                     states: Tuple[torch.Tensor, torch.Tensor] | None) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))

    def get_task_outputs(self, lstm_out: torch.Tensor) -> torch.Tensor:
        """
        Compute outputs for all tasks.

        Args:
            lstm_out: Tensor of shape (seq_len, batch, hidden_dim)
        Returns:
            Tensor of shape (seq_len, batch, n_tasks, hidden_dim)
        """
        return torch.stack([head(lstm_out) for head in self.task_heads], dim=2)

    def log_gate_counts(self) -> None:
        """
        Print a summary of quantum gate counts used during the last forward pass.
        """
        print(f"[QLSTMGen025] Param gates: {self._gate_counts['param_gates']}, "
              f"Entangling gates: {self._gate_counts['entangling_gates']}")

    def _update_gate_counts(self, qlayer: 'QLSTMGen025.QLayer') -> None:
        """
        Update statistics after each quantum gate execution.
        """
        self._gate_counts['param_gates'] += len(qlayer.params)
        self._gate_counts['entangling_gates'] += len(qlayer.cnot_pattern)

class LSTMTaggerGen025(nn.Module):
    """
    Sequence tagging model that can switch between classical and quantum LSTM.
    """
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int,
                 tagset_size: int, n_qubits: int, n_tasks: int = 2,
                 dropout: float = 0.0) -> None:
        if n_qubits <= 0:
            raise ValueError("n_qubits must be > 0 for quantum LSTM.")
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QLSTMGen025(embedding_dim, hidden_dim, n_qubits,
                                n_tasks=n_tasks, dropout=dropout)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor,
                seq_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1),
                                seq_lengths)
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["QLSTMGen025", "LSTMTaggerGen025"]
