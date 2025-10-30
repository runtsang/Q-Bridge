"""Hybrid classifier and tagger with classical feed‑forward and LSTM support."""
from __future__ import annotations

from typing import Iterable, Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridClassifier(nn.Module):
    """
    Hybrid classical classifier/tagger.
    For classification: uses a multi‑layer feed‑forward network.
    For sequence tagging: optionally uses a classical LSTM tagger.
    """
    def __init__(self,
                 num_features: int,
                 depth: int,
                 hidden_dim: int = 128,
                 vocab_size: Optional[int] = None,
                 tagset_size: Optional[int] = None,
                 use_lstm: bool = False):
        super().__init__()
        self.num_features = num_features
        self.depth = depth
        self.use_lstm = use_lstm
        if use_lstm:
            assert vocab_size is not None and tagset_size is not None, \
                "vocab_size and tagset_size must be provided when use_lstm=True"
            self.embedding = nn.Embedding(vocab_size, num_features)
            self.lstm = nn.LSTM(num_features, hidden_dim, batch_first=True)
            self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
            self.task = 'tagging'
        else:
            # Build feed‑forward network
            layers: List[nn.Module] = []
            in_dim = num_features
            for _ in range(depth):
                layers.append(nn.Linear(in_dim, num_features))
                layers.append(nn.ReLU())
                in_dim = num_features
            # head
            layers.append(nn.Linear(in_dim, 2))
            self.network = nn.Sequential(*layers)
            self.task = 'classification'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.task == 'classification':
            return self.network(x)
        else:
            # x: (seq_len, batch, vocab_idx)
            embeds = self.embedding(x)
            lstm_out, _ = self.lstm(embeds)
            logits = self.hidden2tag(lstm_out)
            return F.log_softmax(logits, dim=-1)

    @staticmethod
    def build_classifier_circuit(num_units: int,
                                 depth: int,
                                 *,
                                 use_quantum: bool = False) -> Tuple[object, Iterable, Iterable, List]:
        """
        Return a classical feed‑forward network or a placeholder for quantum circuit.
        For use_quantum=True, returns a dummy object and metadata for compatibility.
        """
        if not use_quantum:
            # Build feed‑forward network
            layers: List[nn.Module] = []
            in_dim = num_units
            encoding = list(range(num_units))
            weight_sizes: List[int] = []
            for _ in range(depth):
                linear = nn.Linear(in_dim, num_units)
                layers.append(linear)
                layers.append(nn.ReLU())
                weight_sizes.append(linear.weight.numel() + linear.bias.numel())
                in_dim = num_units
            head = nn.Linear(in_dim, 2)
            layers.append(head)
            weight_sizes.append(head.weight.numel() + head.bias.numel())
            net = nn.Sequential(*layers)
            observables = list(range(2))
            return net, encoding, weight_sizes, observables
        else:
            # Dummy placeholder to keep interface; actual quantum circuit built in QML module
            return None, list(range(num_units)), [], []
