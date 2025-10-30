import torch
import torch.nn as nn
import torch.nn.functional as F

def build_classifier_circuit(num_features: int, depth: int):
    """
    Classical feed‑forward classifier mirroring the quantum implementation.
    Returns a nn.Sequential network, the list of input indices used for encoding,
    the list of weight‑size counters for each layer, and a placeholder list of
    observable indices.
    """
    layers = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes = []
    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.extend([linear, nn.ReLU(inplace=True)])
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
    Classical implementation of a hybrid LSTM tagger.
    Provides a pure PyTorch LSTM backbone and a classical feed‑forward
    classifier head that is structurally identical to its quantum counterpart.
    """
    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 n_qubits: int = 0,
                 classifier_depth: int = 0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tagset_size = tagset_size
        self.n_qubits = n_qubits
        self.classifier_depth = classifier_depth

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        self.classifier, self.encoding, self.weight_sizes, self.observables = build_classifier_circuit(
            num_features=hidden_dim,
            depth=classifier_depth
        )
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for sequence tagging.
        """
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds)
        tag_logits = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_logits, dim=-1)

    def get_classifier_metadata(self):
        """
        Return metadata used by downstream quantum simulators.
        """
        return self.encoding, self.weight_sizes, self.observables

__all__ = ["HybridQLSTM", "build_classifier_circuit"]
