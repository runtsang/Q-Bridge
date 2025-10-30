import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassicalQLSTM(nn.Module):
    """Classical LSTM cell with linear gates."""
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.forget_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(self, inputs: torch.Tensor, states: tuple | None = None):
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

    def _init_states(self, inputs: torch.Tensor, states: tuple | None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

class ClassicalQuanvolutionFilter(nn.Module):
    """Simple 2‑D convolution acting as a classical quanvolution."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=1, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return features.view(x.size(0), -1)

class HybridQLSTMTagger(nn.Module):
    """Hybrid LSTM tagger that can optionally use a classical quanvolution."""
    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 use_quanvolution: bool = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.use_quanvolution = use_quanvolution
        if self.use_quanvolution:
            self.quanvolution = ClassicalQuanvolutionFilter()
            input_dim = 4 * embedding_dim
        else:
            self.quanvolution = None
            input_dim = embedding_dim
        self.lstm = ClassicalQLSTM(input_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        # sentence shape: (seq_len, batch)
        embeds = self.word_embeddings(sentence)
        if self.quanvolution is not None:
            seq_len, batch, emb_dim = embeds.shape
            x = embeds.view(seq_len * batch, 1, 1, emb_dim)
            features = self.quanvolution(x)
            features = features.view(seq_len, batch, -1)
            lstm_input = features
        else:
            lstm_input = embeds
        lstm_out, _ = self.lstm(lstm_input)
        tag_logits = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_logits, dim=2)
