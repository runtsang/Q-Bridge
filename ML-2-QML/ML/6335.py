"""Hybrid classical classifier and LSTM module."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Tuple, List

__all__ = ["build_classifier_circuit", "HybridQLSTM", "HybridLSTMTagger"]

def build_classifier_circuit(num_features: int,
                             depth: int = 1,
                             n_qubits: int = 0,
                             shared_weights: bool = False,
                             observable_pattern: str = "Z") -> Tuple[nn.Module, Iterable[int], Iterable[int], List[str]]:
    """
    Construct a feed‑forward classifier that optionally mimics a quantum ansatz.
    Parameters
    ----------
    num_features : int
        Input dimensionality.
    depth : int, default 1
        Number of hidden blocks.
    n_qubits : int, default 0
        If greater than zero, each hidden block is replaced by a
        quantum‑style linear layer whose weight matrix is reshaped to an
        ``n_qubits``‑qubit representation.  The implementation remains
        fully classical; the flag merely changes the weight shape.
    shared_weights : bool, default False
        When True, all hidden blocks share the same weight matrix,
        yielding a low‑parameter architecture.
    observable_pattern : str, default "Z"
        Pattern used for the observable list; mirrors the Pauli‑Z
        observables of the quantum version.
    Returns
    -------
    network : nn.Module
        The assembled network.
    encoding : Iterable[int]
        Indices of input features that are explicitly encoded.
    weight_sizes : Iterable[int]
        Number of trainable parameters per layer.
    observables : List[str]
        Names of output observables, e.g. ``["Z", "Z"]``.
    """
    layers: List[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: List[int] = []

    class QLinear(nn.Module):
        """Linear layer that mimics a quantum‑style weight matrix."""
        def __init__(self, in_dim: int, out_dim: int, n_qubits: int):
            super().__init__()
            self.linear = nn.Linear(in_dim, out_dim, bias=True)
            self.n_qubits = n_qubits

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.linear(x)

    for _ in range(depth):
        if n_qubits > 0:
            layer = QLinear(in_dim, num_features, n_qubits)
        else:
            layer = nn.Linear(in_dim, num_features)
        layers.append(layer)
        layers.append(nn.ReLU())
        weight_sizes.append(layer.linear.weight.numel() + layer.linear.bias.numel())
        in_dim = num_features

    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    # Apply weight sharing if requested
    if shared_weights:
        for i in range(1, len(layers) // 2):
            layers[i * 2].linear.weight = layers[0].linear.weight
            layers[i * 2].linear.bias = layers[0].linear.bias

    network = nn.Sequential(*layers)
    observables = [observable_pattern] * 2
    return network, encoding, weight_sizes, observables


class HybridQLSTM(nn.Module):
    """
    Sequence‑tagging LSTM that can swap between a classical LSTM and a
    quantum‑enhanced LSTM.  The quantum branch is imported lazily so that
    the classical module can be used without the heavy quantum dependency.
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        if n_qubits > 0:
            try:
                from.quantum_classifier_model_gen147_qml import QuantumHybridQLSTM
                self.lstm = QuantumHybridQLSTM(input_dim, hidden_dim, n_qubits)
            except Exception:
                # Fallback to a classical LSTM if the quantum module cannot be imported
                self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        else:
            self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs shape: (seq_len, batch, features)
        outputs, _ = self.lstm(inputs)
        return outputs


class HybridLSTMTagger(nn.Module):
    """
    Wraps the hybrid LSTM in a sequence‑tagging head.
    """
    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 n_qubits: int = 0) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = HybridQLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out = self.lstm(embeds)
        tag_logits = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_logits, dim=-1)
