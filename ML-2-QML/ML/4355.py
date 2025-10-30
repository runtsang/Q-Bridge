"""Hybrid LSTM model combining classical, quantum‑inspired gates, autoencoding and fraud‑style regularisation."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the lightweight autoencoder from the seed
try:
    from.Autoencoder import Autoencoder
except Exception:  # pragma: no cover
    # Fallback simple autoencoder if the module is missing
    class Autoencoder(nn.Module):
        def __init__(self, input_dim: int, latent_dim: int = 32):
            super().__init__()
            self.encoder = nn.Linear(input_dim, latent_dim)
        def encode(self, x: torch.Tensor) -> torch.Tensor:
            return self.encoder(x)

class FraudLayer(nn.Module):
    """A compact linear + tanh + affine scaling layer inspired by the photonic fraud detection model."""
    def __init__(self, in_features: int, out_features: int, clip: bool = False) -> None:
        super().__init__()
        weight = torch.randn(out_features, in_features)
        bias = torch.randn(out_features)
        if clip:
            weight.clamp_(-5.0, 5.0)
            bias.clamp_(-5.0, 5.0)
        linear = nn.Linear(in_features, out_features)
        with torch.no_grad():
            linear.weight.copy_(weight)
            linear.bias.copy_(bias)
        self.linear = linear
        self.activation = nn.Tanh()
        self.register_buffer("scale", torch.randn(out_features))
        self.register_buffer("shift", torch.randn(out_features))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(self.linear(x))
        return out * self.scale + self.shift

class ClassicalQLSTMCell(nn.Module):
    """Standard LSTM cell with linear gates."""
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.input_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.forget_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
    def forward(self, x: torch.Tensor, hx: torch.Tensor, cx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([x, hx], dim=1)
        f = torch.sigmoid(self.forget_linear(combined))
        i = torch.sigmoid(self.input_linear(combined))
        g = torch.tanh(self.update_linear(combined))
        o = torch.sigmoid(self.output_linear(combined))
        cx = f * cx + i * g
        hx = o * torch.tanh(cx)
        return hx, cx

class QuantumQLSTMCell(nn.Module):
    """Quantum‑inspired LSTM cell using sin‑based activations to mimic interference."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.input_linear = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.forget_linear = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.update_linear = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.output_linear = nn.Linear(input_dim + hidden_dim, n_qubits)
    def forward(self, x: torch.Tensor, hx: torch.Tensor, cx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([x, hx], dim=1)
        f = torch.sigmoid(torch.sin(self.forget_linear(combined)))
        i = torch.sigmoid(torch.sin(self.input_linear(combined)))
        g = torch.tanh(torch.sin(self.update_linear(combined)))
        o = torch.sigmoid(torch.sin(self.output_linear(combined)))
        cx = f * cx + i * g
        hx = o * torch.tanh(cx)
        return hx, cx

class HybridQLSTM(nn.Module):
    """Drop‑in replacement that can use classical, quantum‑inspired gates and a lightweight autoencoder."""
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int = 0,
        use_autoencoder: bool = False,
        autoencoder_latent_dim: int = 32,
        fraud_clip: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.use_autoencoder = use_autoencoder

        if use_autoencoder:
            self.autoencoder = Autoencoder(input_dim, latent_dim=autoencoder_latent_dim)
            self.input_dim = autoencoder_latent_dim

        if n_qubits > 0:
            self.lstm_cell = QuantumQLSTMCell(self.input_dim, hidden_dim, n_qubits)
        else:
            self.lstm_cell = ClassicalQLSTMCell(self.input_dim, hidden_dim)

        self.fraud_layer = FraudLayer(2, 2, clip=fraud_clip)
        self.output_linear = nn.Linear(hidden_dim, 1)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of shape (seq_len, batch, input_dim).

        Returns
        -------
        out : torch.Tensor
            Final predictions of shape (seq_len, batch, 1).
        states : Tuple[torch.Tensor, torch.Tensor]
            Final hidden and cell states.
        """
        hx = torch.zeros(inputs.size(1), self.hidden_dim, device=inputs.device)
        cx = torch.zeros(inputs.size(1), self.hidden_dim, device=inputs.device)
        outputs = []

        for t in range(inputs.size(0)):
            x = inputs[t]
            if self.use_autoencoder:
                x = self.autoencoder.encode(x)
            hx, cx = self.lstm_cell(x, hx, cx)
            hx = self.fraud_layer(hx)
            outputs.append(hx.unsqueeze(0))

        outputs = torch.cat(outputs, dim=0)
        out = self.output_linear(outputs)
        return out, (hx, cx)

class LSTMTagger(nn.Module):
    """Sequence tagging model that uses :class:`HybridQLSTM`."""
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        use_autoencoder: bool = False,
        autoencoder_latent_dim: int = 32,
        fraud_clip: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = HybridQLSTM(
            embedding_dim,
            hidden_dim,
            n_qubits=n_qubits,
            use_autoencoder=use_autoencoder,
            autoencoder_latent_dim=autoencoder_latent_dim,
            fraud_clip=fraud_clip,
        )
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["HybridQLSTM", "LSTMTagger", "FraudLayer", "ClassicalQLSTMCell", "QuantumQLSTMCell"]
