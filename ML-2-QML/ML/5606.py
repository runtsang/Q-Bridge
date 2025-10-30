import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassicalQLSTM(nn.Module):
    """Classical LSTM cell with linear gates, used as a fallback when no qubits are specified."""
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

    def forward(self, inputs: torch.Tensor, states: tuple[torch.Tensor, torch.Tensor] | None = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in torch.unbind(inputs, dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_linear(combined))
            i = torch.sigmoid(self.input_linear(combined))
            g = torch.tanh(self.update_linear(combined))
            o = torch.sigmoid(self.output_linear(combined))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(self, inputs: torch.Tensor, states: tuple[torch.Tensor, torch.Tensor] | None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx


def EstimatorQNN(input_dim: int = 2) -> nn.Module:
    """Simple feed‑forward regressor mirroring the quantum EstimatorQNN."""
    class EstimatorNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 8),
                nn.Tanh(),
                nn.Linear(8, 4),
                nn.Tanh(),
                nn.Linear(4, 1),
            )

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return self.net(inputs)

    return EstimatorNN()


def SamplerQNN(input_dim: int = 2) -> nn.Module:
    """Simple sampler network mirroring the quantum SamplerQNN."""
    class SamplerModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 4),
                nn.Tanh(),
                nn.Linear(4, 2),
            )

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return F.softmax(self.net(inputs), dim=-1)

    return SamplerModule()


def build_classifier_circuit(num_features: int, depth: int) -> tuple[nn.Module, list[int], list[int], list[int]]:
    """Construct a classical feed‑forward classifier matching the quantum interface."""
    layers = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes = []
    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.extend([linear, nn.ReLU()])
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features
    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())
    network = nn.Sequential(*layers)
    observables = list(range(2))
    return network, encoding, weight_sizes, observables


class HybridQLSTM(nn.Module):
    """
    Hybrid LSTM that can operate in classical or quantum mode, enriches hidden states with
    a quantum estimator and sampler, and feeds the combined representation to a classifier.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        use_estimator: bool = True,
        use_sampler: bool = True,
        use_classifier: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        if n_qubits > 0:
            self.lstm = ClassicalQLSTM(embedding_dim, hidden_dim, n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=False)

        self.use_estimator = use_estimator
        self.use_sampler = use_sampler
        self.use_classifier = use_classifier

        if use_estimator:
            self.estimator = EstimatorQNN(hidden_dim)
        if use_sampler:
            self.sampler = SamplerQNN(hidden_dim)
        if use_classifier:
            feature_dim = hidden_dim
            if use_estimator:
                feature_dim += 1
            if use_sampler:
                feature_dim += 2
            self.classifier, _, _, _ = build_classifier_circuit(feature_dim, depth=2)

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        sentence : torch.Tensor
            Tensor of shape (seq_len, batch) containing word indices.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            log‑softmax logits from the classifier and the tagger.
        """
        embeds = self.word_embeddings(sentence)  # (seq_len, batch, embed)
        lstm_out, _ = self.lstm(embeds)
        # Use last hidden state for downstream modules
        hidden_last = lstm_out[-1]  # (batch, hidden)
        features = hidden_last

        if self.use_estimator:
            est = self.estimator(features)  # (batch, 1)
            features = torch.cat([features, est], dim=1)
        if self.use_sampler:
            samp = self.sampler(features)  # (batch, 2)
            features = torch.cat([features, samp], dim=1)
        if self.use_classifier:
            logits = self.classifier(features)  # (batch, 2)
            tag_logits = self.hidden2tag(lstm_out[-1])  # (batch, tagset_size)
            return F.log_softmax(logits, dim=1), F.log_softmax(tag_logits, dim=1)
        else:
            return F.log_softmax(hidden_last, dim=1), None
