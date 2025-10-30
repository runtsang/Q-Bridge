"""Hybrid classical LSTM with quantum‑style preprocessing and attention.

The class ``QLSTM`` is a pure PyTorch module that delegates the core
recurrent update to a quantum implementation (``QuantumQLSTM``) but
remains fully compatible with existing training pipelines.  The
wrapper also incorporates a quantum convolution pre‑processor and a
quantum self‑attention post‑processor, both of which expose a simple
``run`` method that returns classical tensors.

The accompanying ``LSTMTagger`` demonstrates how the hybrid cell
fits into a sequence‑tagging task."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Quantum modules live in the companion ``qlstm_qml`` package
from qlstm_qml import QLSTM as QuantumQLSTM
from qlstm_qml import QuantumConv, QuantumSelfAttention


class QLSTM(nn.Module):
    """Classical LSTM cell that uses a quantum‑gated core and quantum
    preprocessing/post‑processing steps.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input features.
    hidden_dim : int
        Hidden state dimensionality.
    n_qubits : int
        Number of qubits used in the quantum LSTM gates.
    n_conv : int, optional
        Size of the quantum convolution kernel (default ``2``).
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int,
                 n_conv: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Quantum convolution pre‑processor
        self.conv = QuantumConv(kernel_size=n_conv)

        # Quantum‑gated LSTM core
        self.quantum_lstm = QuantumQLSTM(input_dim, hidden_dim, n_qubits)

        # Quantum self‑attention post‑processor
        self.attn = QuantumSelfAttention(n_qubits=n_qubits)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of shape ``(seq_len, batch, embed_dim)``.

        Returns
        -------
        torch.Tensor
            Attended output of shape ``(seq_len, batch, hidden_dim)``.
        """
        seq_len, batch, _ = inputs.shape

        # 1. Quantum convolution on each time step
        conv_out = []
        for t in range(seq_len):
            # ``run`` returns a scalar tensor
            conv_out.append(self.conv.run(inputs[t].detach().cpu().numpy()))
        conv_out = torch.stack(conv_out, dim=0)  # shape (seq_len, 1)

        # 2. Quantum LSTM core
        lstm_out, _ = self.quantum_lstm(conv_out)

        # 3. Quantum self‑attention over the hidden states
        # Generate random parameters for the attention circuit
        rot_params = torch.rand(12).numpy()      # 4 qubits * 3 angles
        ent_params = torch.rand(3).numpy()       # 4 qubits - 1 entangle params
        attn_out = self.attn.run(
            rotation_params=rot_params,
            entangle_params=ent_params,
            inputs=lstm_out.detach().cpu().numpy()
        )
        attn_out = torch.tensor(attn_out, device=inputs.device)

        return attn_out


class LSTMTagger(nn.Module):
    """Sequence tagging model that can switch between a classical
    ``nn.LSTM`` and the hybrid quantum‑classical ``QLSTM``."""
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int,
                 tagset_size: int, n_qubits: int = 0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["QLSTM", "LSTMTagger"]
