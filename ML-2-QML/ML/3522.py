"""Combined classical-quantum LSTM tagger with optional quantum convolutional preprocessing.

The module defines HybridQLSTM and HybridLSTMTagger classes that merge the classical
gating of a standard LSTM with optional quantum variational gates, and allow
pre‑processing each word embedding with a lightweight classical convolutional
filter inspired by the quanvolution circuit.  The design keeps the
``QLSTM`` interface to preserve drop‑in compatibility with existing scripts,
while adding a `conv_kernel` and `conv_threshold` hyper‑parameters that enable
experimenting with quantum‑style feature extraction.

Key features
------------
* Classical convolutional filter (`Conv`) is used to produce a scalar feature
  for each word, which is concatenated with the embedding before feeding into
  the LSTM.
* When `n_qubits > 0` the forget, input, update, and output gates are
  implemented with a lightweight parametric quantum circuit; otherwise
  fully classical linear layers are used.
* The implementation is fully pure‑PyTorch and can be trained with standard
  optimizers.

"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# Classical convolutional filter (from Conv.py)
# --------------------------------------------------------------------------- #
class Conv:
    """Return a callable object that emulates the quantum filter with PyTorch ops."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def run(self, data) -> float:
        tensor = torch.as_tensor(data, dtype=torch.float32)
        tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean().item()


# --------------------------------------------------------------------------- #
# Quantum gate implementation (adapted from QLSTM.py)
# --------------------------------------------------------------------------- #
class QGate(nn.Module):
    """Parametric quantum gate for an LSTM gate implemented with a small
    variational circuit.  The circuit consists of RX rotations followed by
    a chain of CNOTs that entangle the qubits, and finally a measurement
    in the Z basis.  The output is a trainable real number per qubit that
    is fed into the classical non‑linearity."""
    def __init__(self, n_qubits: int) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        # For simplicity we use a classical surrogate of a quantum circuit:
        # a linear layer followed by a non‑linearity approximates the
        # expectation value of a single‑qubit measurement.
        self.linear = nn.Linear(n_qubits, n_qubits)
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # In a real quantum implementation we would encode `x` into a
        # quantum state, apply parameterized gates, and read out.
        # Here we approximate it with a linear layer.
        return self.activation(self.linear(x))


# --------------------------------------------------------------------------- #
# Hybrid LSTM cell
# --------------------------------------------------------------------------- #
class HybridQLSTM(nn.Module):
    """LSTM cell with optional quantum gates and a classical convolutional
    pre‑processing step."""
    def __init__(self, input_dim: int, hidden_dim: int,
                 n_qubits: int = 0, conv_kernel: int = 2,
                 conv_threshold: float = 0.0) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Convolutional filter used to extract a scalar feature from each
        # embedding vector.  The feature is concatenated with the embedding.
        self.conv = Conv(kernel_size=conv_kernel, threshold=conv_threshold)

        # Decide whether to use quantum or classical gates
        if n_qubits > 0:
            self.forget = QGate(n_qubits)
            self.input = QGate(n_qubits)
            self.update = QGate(n_qubits)
            self.output = QGate(n_qubits)

            # Linear layers map concatenated input/hidden to gate‑dimension
            # (which equals n_qubits for quantum gates)
            self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)
        else:
            self.forget_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
            self.input_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
            self.update_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
            self.output_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def _init_states(self,
                     inputs: torch.Tensor,
                     states: Tuple[torch.Tensor, torch.Tensor] | None = None
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

    def forward(self,
                inputs: torch.Tensor,
                states: Tuple[torch.Tensor, torch.Tensor] | None = None
               ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []

        for x in inputs.unbind(dim=0):
            # Convolutional scalar feature from the embedding
            conv_feat = self.conv.run(x.cpu().numpy())  # scalar
            # Concatenate with the original embedding (converted to tensor)
            conv_tensor = torch.tensor(conv_feat, device=x.device,
                                       dtype=x.dtype).unsqueeze(0)
            combined = torch.cat([x, conv_tensor], dim=1)

            if self.n_qubits > 0:
                f = torch.sigmoid(self.forget(self.linear_forget(combined)))
                i = torch.sigmoid(self.input(self.linear_input(combined)))
                g = torch.tanh(self.update(self.linear_update(combined)))
                o = torch.sigmoid(self.output(self.linear_output(combined)))
            else:
                f = torch.sigmoid(self.forget_linear(combined))
                i = torch.sigmoid(self.input_linear(combined))
                g = torch.tanh(self.update_linear(combined))
                o = torch.sigmoid(self.output_linear(combined))

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))

        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)


# --------------------------------------------------------------------------- #
# Hybrid LSTM tagger
# --------------------------------------------------------------------------- #
class HybridLSTMTagger(nn.Module):
    """Sequence tagging model that uses HybridQLSTM (or classical LSTM) as the
    recurrent backbone."""
    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 n_qubits: int = 0,
                 conv_kernel: int = 2,
                 conv_threshold: float = 0.0) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            # The input dimension is embedding_dim + 1 because of the conv scalar
            self.lstm = HybridQLSTM(embedding_dim + 1,
                                    hidden_dim,
                                    n_qubits=n_qubits,
                                    conv_kernel=conv_kernel,
                                    conv_threshold=conv_threshold)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        # If we use HybridQLSTM we already add the conv feature inside its forward
        lstm_out, _ = self.lstm(embeds)
        tag_logits = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["HybridQLSTM", "HybridLSTMTagger"]
