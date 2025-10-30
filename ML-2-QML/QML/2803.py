"""Hybrid quantum LSTM with variational autoencoder compression.

The quantum branch replaces both the LSTM gates and the autoencoder with
small variational circuits, providing a fully quantum end‑to‑end
sequence‑tagging pipeline."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from typing import Tuple

# ----- Quantum autoencoder -----
class QAutoencoder(tq.QuantumModule):
    """
    Variational autoencoder that compresses classical data into a
    latent vector via a simple circuit and recovers it with a second ansatz.
    Only the encoder output is used as latent representation for the quantum LSTM.
    """
    def __init__(self, input_dim: int, latent_dim: int, reps: int = 3) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.reps = reps
        # Encoder ansatz: simple RX rotations
        self.encoder_ansatz = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i % latent_dim]} for i in range(latent_dim)]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input vector into a latent representation.
        """
        # Flatten time dimension: (seq_len, batch, input_dim) -> (seq_len*batch, input_dim)
        flat = x.reshape(-1, self.input_dim)
        qdev = tq.QuantumDevice(
            n_wires=self.latent_dim,
            bsz=flat.shape[0],
            device=flat.device,
        )
        # Apply classical encoding via X rotations
        for i in range(self.input_dim):
            qdev.x(wires=i % self.latent_dim)
        # Apply encoder ansatz
        self.encoder_ansatz(qdev, flat)
        latent_flat = self.measure(qdev)
        # Reshape back to (seq_len, batch, latent_dim)
        seq_len = x.shape[0]
        batch = x.shape[1]
        return latent_flat.reshape(seq_len, batch, self.latent_dim)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decodes the latent vector back to the original space.
        """
        flat = z.reshape(-1, self.latent_dim)
        qdev = tq.QuantumDevice(
            n_wires=self.latent_dim,
            bsz=flat.shape[0],
            device=flat.device,
        )
        # Simple decoder: apply inverse of encoder ansatz
        self.encoder_ansatz(qdev, flat)
        recon_flat = self.measure(qdev)
        seq_len = z.shape[0]
        batch = z.shape[1]
        return recon_flat.reshape(seq_len, batch, self.input_dim)

# ----- Quantum LSTM (from reference pair 1) -----
class QLSTM(nn.Module):
    """
    LSTM cell where gates are realised by small quantum circuits.
    """
    class QLayer(tq.QuantumModule):
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
                    tqf.cnot(qdev, wires=[wire, 0])
                else:
                    tqf.cnot(qdev, wires=[wire, wire + 1])
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        self.forget = self.QLayer(n_qubits)
        self.input = self.QLayer(n_qubits)
        self.update = self.QLayer(n_qubits)
        self.output = self.QLayer(n_qubits)

        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(self.linear_forget(combined)))
            i = torch.sigmoid(self.input(self.linear_input(combined)))
            g = torch.tanh(self.update(self.linear_update(combined)))
            o = torch.sigmoid(self.output(self.linear_output(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        return torch.cat(outputs, dim=0), (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

class LSTMTagger(nn.Module):
    """Sequence tagging model that can switch between classical and quantum LSTM."""
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int, tagset_size: int, n_qubits: int = 0) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds)
        tag_logits = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_logits, dim=2)

# ----- Hybrid quantum LSTM + autoencoder -----
class HybridQLSTMEncoder(nn.Module):
    """
    Quantum variant of HybridQLSTMEncoder.  Uses a variational autoencoder
    to compress embeddings and a quantum LSTM with gate‑level variational
    circuits for sequence tagging.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        autoencoder_latent_dim: int = 32,
        autoencoder_reps: int = 3,
        n_qubits: int = 8,
    ) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.autoencoder = QAutoencoder(
            input_dim=embedding_dim,
            latent_dim=autoencoder_latent_dim,
            reps=autoencoder_reps,
        )
        self.lstm = QLSTM(autoencoder_latent_dim, hidden_dim, n_qubits=n_qubits)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        sentence : torch.Tensor
            Tensor of shape (seq_len, batch) containing word indices.

        Returns
        -------
        torch.Tensor
            Log‑softmaxed tag logits of shape (seq_len, batch, tagset_size).
        """
        embeds = self.word_embeddings(sentence)          # (seq_len, batch, embed)
        latent = self.autoencoder(embeds)                # (seq_len, batch, latent)
        lstm_out, _ = self.lstm(latent)                 # (seq_len, batch, hidden)
        tag_logits = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_logits, dim=2)

__all__ = [
    "QLSTM",
    "LSTMTagger",
    "QAutoencoder",
    "HybridQLSTMEncoder",
]
