import torch
import torch.nn as nn
import torch.nn.functional as F
from FraudDetection import build_fraud_detection_program, FraudLayerParameters

class QLSTM(nn.Module):
    """Drop‑in classical LSTM cell that mimics the quantum interface."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.forget_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)

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
        return torch.cat(outputs, dim=0), (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

class HybridTagger(nn.Module):
    """
    Sequence tagging model that optionally uses a quantum LSTM
    and includes a fraud‑detection sub‑module.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        fraud_params: FraudLayerParameters | None = None,
        fraud_layers: list[FraudLayerParameters] | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        if fraud_params is not None and fraud_layers is not None:
            self.fraud_net = build_fraud_detection_program(fraud_params, fraud_layers)
        else:
            self.fraud_net = None

    def forward(self, sentence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_logits = F.log_softmax(tag_logits, dim=1)
        fraud_output = None
        if self.fraud_net is not None:
            # Use the last embedding as the fraud feature vector
            fraud_input = embeds[-1].unsqueeze(0)  # shape (1, embedding_dim)
            if fraud_input.size(1) > 2:
                fraud_input = fraud_input[:, :2]
            fraud_output = self.fraud_net(fraud_input)
        return tag_logits, fraud_output

__all__ = ["HybridTagger", "QLSTM"]
