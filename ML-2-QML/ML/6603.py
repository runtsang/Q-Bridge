import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridQLSTM(nn.Module):
    """
    Hybrid LSTM cell that augments classical gates with a quantum‑inspired
    feature map.  The API matches the original QLSTM so it can be used
    interchangeably in downstream models.
    """

    def __init__(self, input_dim: int, hidden_dim: int,
                 n_qubits: int = 4, sparsity: float = 0.2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Classical LSTM gates
        self.forget_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)

        # Quantum‑inspired feature map
        self.qmap = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.qact = nn.Tanh()

        # Attention‑like fusion of classical and quantum features
        self.fusion = nn.Linear(n_qubits + hidden_dim, hidden_dim)

        # Scaling coefficient for quantum contribution
        self.alpha = nn.Parameter(torch.tensor(0.5))

        # Sparsity mask for the quantum feature map
        self.register_buffer(
           'mask',
            torch.bernoulli(torch.full((n_qubits,), 1 - sparsity)).float()
        )

    def _init_states(self, inputs, states):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

    def forward(self, inputs: torch.Tensor,
                states: tuple | None = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_gate(combined))
            i = torch.sigmoid(self.input_gate(combined))
            g = torch.tanh(self.update_gate(combined))
            o = torch.sigmoid(self.output_gate(combined))

            # Quantum‑inspired feature
            qfeat_raw = self.qmap(combined)
            qfeat = self.qact(qfeat_raw)
            qfeat = qfeat * self.mask

            # Fuse with classical hidden state
            fused = torch.cat([hx, qfeat], dim=1)
            fused = self.fusion(fused)
            fused = fused * self.alpha

            cx = f * cx + i * g
            hx = o * torch.tanh(cx) + fused
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

class LSTMTagger(nn.Module):
    """
    Sequence tagging model that can use either the hybrid
    quantum‑classical LSTM or a vanilla nn.LSTM.
    """

    def __init__(self, embedding_dim: int, hidden_dim: int,
                 vocab_size: int, tagset_size: int,
                 n_qubits: int = 0, sparsity: float = 0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = HybridQLSTM(embedding_dim, hidden_dim,
                                    n_qubits=n_qubits,
                                    sparsity=sparsity)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["HybridQLSTM", "LSTMTagger"]
