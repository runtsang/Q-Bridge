"""Hybrid LSTM with attention and autoencoding – quantum implementation.

The quantum module replaces all linear layers with variational circuits
and uses a quantum self‑attention block together with a quantum autoencoder.
It retains the same public interface as the classical version, enabling
side‑by‑side benchmarking.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchquantum as tq
from typing import Tuple, List, Optional


# --------------------------------------------------------------------------- #
#  Quantum self‑attention – variational circuit
# --------------------------------------------------------------------------- #
class QuantumSelfAttention(tq.QuantumModule):
    """Quantum self‑attention block that learns rotation and entanglement parameters.

    The circuit encodes the input vector into qubits, applies a trainable
    rotation layer, entangles adjacent qubits, and measures all qubits.
    The measurement outcomes are used to compute a soft‑max attention
    distribution over the sequence.
    """

    def __init__(self, n_wires: int) -> None:
        super().__init__()
        self.n_wires = n_wires
        # Parameterised rotation layer
        self.rotation_params = nn.Parameter(torch.randn(n_wires, n_wires))
        # Parameterised entanglement (controlled‑X) strengths
        self.entangle_params = nn.Parameter(torch.randn(n_wires, n_wires))
        # Encoder to map classical data into qubit amplitudes
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{n_wires}xRy"])
        # Measurement
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
        # Encode classical input
        self.encoder(qdev, qdev.state)
        # Apply learnable rotations
        for i in range(self.n_wires):
            for j in range(self.n_wires):
                tq.RX(self.rotation_params[i, j], has_params=True, trainable=True)(qdev, wires=i)
        # Entangle adjacent qubits
        for i in range(self.n_wires - 1):
            tq.CNOT(qdev, wires=[i, i + 1])
        # Measure all qubits
        return self.measure(qdev)


# --------------------------------------------------------------------------- #
#  Quantum autoencoder – variational circuit
# --------------------------------------------------------------------------- #
class QuantumAutoencoder(tq.QuantumModule):
    """Quantum autoencoder that maps a state to a latent vector and back."""

    class _QLayer(tq.QuantumModule):
        def __init__(self, num_wires: int) -> None:
            super().__init__()
            self.n_wires = num_wires
            self.random_layer = tq.RandomLayer(n_ops=20, wires=list(range(num_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random_layer(qdev)
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)

    def __init__(self, num_wires: int) -> None:
        super().__init__()
        self.n_wires = num_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self._QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)


# --------------------------------------------------------------------------- #
#  Quantum LSTM cell – gates realised by small quantum circuits
# --------------------------------------------------------------------------- #
class QuantumQLSTM(tq.QuantumModule):
    """Quantum LSTM cell where each gate is a small variational circuit."""

    class _QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [0], "func": "rx", "wires": [0]},
                    {"input_idx": [1], "func": "rx", "wires": [1]},
                    {"input_idx": [2], "func": "rx", "wires": [2]},
                    {"input_idx": [3], "func": "rx", "wires": [3]},
                ]
            )
            self.params = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            for wire in range(self.n_wires):
                if wire == self.n_wires - 1:
                    tq.cnot(qdev, wires=[wire, 0])
                else:
                    tq.cnot(qdev, wires=[wire, wire + 1])
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        self.forget = self._QLayer(n_qubits)
        self.input = self._QLayer(n_qubits)
        self.update = self._QLayer(n_qubits)
        self.output = self._QLayer(n_qubits)

        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(self, inputs: torch.Tensor,
                states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs: List[torch.Tensor] = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(self.linear_forget(combined)))
            i = torch.sigmoid(self.input(self.linear_input(combined)))
            g = torch.tanh(self.update(self.linear_update(combined)))
            o = torch.sigmoid(self.output(self.linear_output(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def _init_states(self, inputs: torch.Tensor,
                     states: Optional[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx


# --------------------------------------------------------------------------- #
#  Hybrid quantum model – combines quantum LSTM, attention, autoencoder and regression
# --------------------------------------------------------------------------- #
class HybridQLSTM(tq.QuantumModule):
    """Quantum‑enabled hybrid LSTM that mirrors the classical interface.

    The model can be instantiated either with a purely classical LSTM
    (``n_qubits=0``) or with the quantum LSTM cell (``n_qubits>0``).  It
    includes a quantum self‑attention block and a quantum autoencoder,
    followed by a regression head that operates on the latent representation.
    """

    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 latent_dim: int = 32,
                 n_qubits: int = 0) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tagset_size = tagset_size
        self.latent_dim = latent_dim
        self.n_qubits = n_qubits

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QuantumQLSTM(embedding_dim, hidden_dim, n_qubits) if n_qubits > 0 else nn.LSTM(embedding_dim, hidden_dim)
        self.attention = QuantumSelfAttention(n_qubits)
        self.autoencoder = QuantumAutoencoder(n_qubits)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.regressor = nn.Linear(latent_dim, 1)

    def forward(self, sentence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        sentence : torch.Tensor
            Token indices of shape (seq_len, batch)

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            * Tag logits (log‑softmax) of shape (seq_len, tagset_size)
            * Regression scalar of shape (1,)
        """
        embeds = self.word_embeddings(sentence)  # (seq_len, batch, embed)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))  # (seq_len, batch, hidden)

        # Attention – we use the quantum attention block on the LSTM output
        # For simplicity we collapse the batch dimension and feed into the circuit
        qdev = tq.QuantumDevice(n_wires=self.n_qubits, bsz=1, device=embeds.device)
        # Use the first token as a simple example; in practice you would loop over the sequence
        qdev.state = lstm_out.squeeze(1)[0].unsqueeze(0)
        attn_meas = self.attention(qdev)  # (1, n_qubits)
        attn_tensor = attn_meas.squeeze(0)  # (n_qubits,)

        # Autoencoder – feed the attention measurement as a state
        latent = self.autoencoder(attn_tensor.unsqueeze(0))  # (1,)

        # Regression: mean over sequence (here we use the single latent value)
        reg_output = self.regressor(latent)  # (1,)

        # Tagging head
        tag_logits = self.hidden2tag(lstm_out.squeeze(1))  # (seq_len, tagset)

        return F.log_softmax(tag_logits, dim=1), reg_output.squeeze(0)

__all__ = ["HybridQLSTM", "QuantumQLSTM", "QuantumSelfAttention", "QuantumAutoencoder"]
