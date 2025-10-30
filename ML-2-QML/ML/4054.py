import torch
import torch.nn as nn
import torch.nn.functional as F

class SamplerQNN(nn.Module):
    """Classical sampler mirroring the QNN helper."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(inputs), dim=-1)

class RegressionHead(nn.Module):
    """Simple regression head used when the model is in regression mode."""
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

class QLSTM(nn.Module):
    """Classical LSTM cell with linear gates."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        gate_dim = hidden_dim
        self.forget_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, gate_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None = None,
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

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

class HybridQLSTM(nn.Module):
    """
    A hybrid LSTM that can operate in classical or quantum mode,
    optionally performing sequence tagging or regression.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        regression: bool = False,
        regression_input_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.tagging = not regression
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # LSTM backbone
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # Output heads
        if self.tagging:
            self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        else:
            self.regression_head = RegressionHead(regression_input_dim or hidden_dim)

        # Sampler for initial hidden state
        self.sampler = SamplerQNN()

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        # The sampler is used only as a placeholder for generating initial states.
        _ = self.sampler(torch.rand(1, 2).to(sentence.device))

        embeds = self.word_embeddings(sentence)
        lstm_out, (hx, cx) = self.lstm(embeds.view(len(sentence), 1, -1))

        if self.tagging:
            logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
            return F.log_softmax(logits, dim=1)
        else:
            return self.regression_head(hx.squeeze(0))

__all__ = ["HybridQLSTM", "QLSTM", "SamplerQNN", "RegressionHead"]
