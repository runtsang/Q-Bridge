import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridActivation(torch.autograd.Function):
    """Differentiable sigmoid with a learnable shift, emulating a quantum expectation head."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float):
        outputs = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (outputs,) = ctx.saved_tensors
        grad_inputs = grad_output * outputs * (1 - outputs)
        return grad_inputs, None

class HybridGate(nn.Module):
    """Linear gate followed by a hybrid activation."""
    def __init__(self, in_features: int, out_features: int, shift: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x)
        return HybridActivation.apply(logits, self.shift)

class QLSTM(nn.Module):
    """Classical LSTM where the gates are realized by HybridGate modules."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int, shift: float = 0.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.shift = shift

        self.forget_gate = HybridGate(input_dim + hidden_dim, n_qubits, shift)
        self.input_gate  = HybridGate(input_dim + hidden_dim, n_qubits, shift)
        self.update_gate = HybridGate(input_dim + hidden_dim, n_qubits, shift)
        self.output_gate = HybridGate(input_dim + hidden_dim, n_qubits, shift)

        self.forget_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.input_lin  = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.update_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.output_lin = nn.Linear(input_dim + hidden_dim, n_qubits)

    def _init_states(self, inputs: torch.Tensor, states):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

    def forward(self, inputs: torch.Tensor, states=None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = self.forget_gate(self.forget_lin(combined))
            i = self.input_gate(self.input_lin(combined))
            g = torch.tanh(self.update_gate(self.update_lin(combined)))
            o = self.output_gate(self.output_lin(combined))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

class LSTMTagger(nn.Module):
    """Sequence tagging model that can use either the hybrid QLSTM or a vanilla LSTM."""
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int,
                 tagset_size: int, n_qubits: int = 0, shift: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits, shift)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["QLSTM", "LSTMTagger"]
