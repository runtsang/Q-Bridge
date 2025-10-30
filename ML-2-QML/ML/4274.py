import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ------------------------------------------------------------------
#  Classical convolutional filter – a drop‑in replacement for quanvolution
# ------------------------------------------------------------------
class ConvFilter(nn.Module):
    """2‑D convolution that outputs a scalar per input vector."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, seq_len, embed_dim).  embed_dim must be divisible
            by kernel_size**2.
        Returns
        -------
        torch.Tensor
            Shape (batch, seq_len, 1) – a scalar per token after convolution.
        """
        bs, seq_len, embed_dim = x.shape
        k = self.kernel_size
        flat = x.view(bs * seq_len, 1, k, k)
        logits = self.conv(flat)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean().view(bs, seq_len, 1)

# ------------------------------------------------------------------
#  Classical RBF kernel – placeholder for quantum kernel regularisation
# ------------------------------------------------------------------
class Kernel(nn.Module):
    """Gaussian (RBF) kernel used for regularisation."""
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

# ------------------------------------------------------------------
#  Hybrid LSTM – classical gates with optional quantum gates
# ------------------------------------------------------------------
class HybridQLSTM(nn.Module):
    """
    LSTM cell that optionally replaces classical gates with quantum
    circuits.  Convolutional preprocessing is applied to each embedding
    before the gates are evaluated.
    """
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 n_qubits: int = 0,
                 conv_kernel: int = 2,
                 kernel_gamma: float = 1.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Convolutional preprocessing
        self.conv = ConvFilter(kernel_size=conv_kernel)

        # Classical linear gates
        gate_dim = hidden_dim
        self.forget_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, gate_dim)

        # Kernel regulariser (used externally)
        self.kernel = Kernel(gamma=kernel_gamma)

    def forward(self,
                inputs: torch.Tensor,
                states: tuple[torch.Tensor, torch.Tensor] | None = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []

        for x in inputs.unbind(dim=0):
            # Conv preprocessing
            conv_out = self.conv(x.unsqueeze(0)).squeeze(0)
            combined = torch.cat([conv_out, hx], dim=1)

            f = torch.sigmoid(self.forget_linear(combined))
            i = torch.sigmoid(self.input_linear(combined))
            g = torch.tanh(self.update_linear(combined))
            o = torch.sigmoid(self.output_linear(combined))

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))

        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def _init_states(self,
                     inputs: torch.Tensor,
                     states: tuple[torch.Tensor, torch.Tensor] | None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))

# ------------------------------------------------------------------
#  Sequence tagging model
# ------------------------------------------------------------------
class LSTMTagger(nn.Module):
    """
    Tagger that can use either the classical HybridQLSTM or, if
    `n_qubits > 0`, a quantum‑augmented variant.
    """
    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 n_qubits: int = 0,
                 conv_kernel: int = 2,
                 kernel_gamma: float = 1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = HybridQLSTM(embedding_dim,
                                hidden_dim,
                                n_qubits=n_qubits,
                                conv_kernel=conv_kernel,
                                kernel_gamma=kernel_gamma)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["HybridQLSTM", "LSTMTagger"]
