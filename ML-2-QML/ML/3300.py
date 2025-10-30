import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ClassicalQLSTM(nn.Module):
    """Classical LSTM cell mirroring the original QLSTM interface."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        gate_dim = hidden_dim
        self.forget_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, gate_dim)

    def forward(self, inputs: torch.Tensor,
                states: tuple[torch.Tensor, torch.Tensor] | None = None
                ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_linear(combined))
            i = torch.sigmoid(self.input_linear(combined))
            g = torch.tanh(self.update_linear(combined))
            o = torch.sigmoid(self.output_linear(combined))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def _init_states(self, inputs: torch.Tensor,
                     states: tuple[torch.Tensor, torch.Tensor] | None
                     ) -> tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx


class ClassicalSelfAttention:
    """NumPy‑based self‑attention compatible with the quantum interface."""
    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim

    def run(self, rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            inputs: np.ndarray) -> np.ndarray:
        query = torch.as_tensor(inputs @ rotation_params.reshape(self.embed_dim, -1),
                                dtype=torch.float32)
        key = torch.as_tensor(inputs @ entangle_params.reshape(self.embed_dim, -1),
                              dtype=torch.float32)
        value = torch.as_tensor(inputs, dtype=torch.float32)
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return (scores @ value).numpy()


class LSTMTagger(nn.Module):
    """Tagger that can operate in classical mode with optional self‑attention."""
    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 n_qubits: int = 0,
                 use_attention: bool = True) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # In the pure classical build n_qubits is ignored; the flag is kept for API parity.
        self.lstm = ClassicalQLSTM(embedding_dim, hidden_dim, n_qubits=hidden_dim)

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.use_attention = use_attention
        if use_attention:
            self.attention = ClassicalSelfAttention(embedding_dim)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))

        if self.use_attention:
            # Random parameters are placeholders; in practice these would be learned.
            rotation_params = np.random.randn(self.hidden_dim, self.hidden_dim)
            entangle_params = np.random.randn(self.hidden_dim, self.hidden_dim)
            attn_out = self.attention.run(rotation_params,
                                          entangle_params,
                                          lstm_out.detach().cpu().numpy())
            attn_tensor = torch.tensor(attn_out,
                                       device=lstm_out.device,
                                       dtype=lstm_out.dtype)
            lstm_out = lstm_out + attn_tensor

        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)
