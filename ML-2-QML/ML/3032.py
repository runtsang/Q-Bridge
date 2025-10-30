import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional

class UnifiedClassifier(nn.Module):
    """
    Unified classifier that supports classical MLP, quantum data‑uploading circuit,
    and an optional quantum‑enhanced LSTM for sequence tagging.
    The interface mirrors the original QuantumClassifierModel, enabling
    drop‑in replacement while exposing a tunable depth and a quantum flag.
    """
    def __init__(self,
                 num_features: int,
                 depth: int = 1,
                 mode: str = "classical",
                 n_qubits: Optional[int] = None,
                 seq_mode: bool = False,
                 seq_hidden_dim: int = 128,
                 vocab_size: int = 0,
                 tagset_size: int = 0,
                 quantum_circuit_builder: Optional[Callable] = None):
        super().__init__()
        self.num_features = num_features
        self.depth = depth
        self.mode = mode
        self.n_qubits = n_qubits if n_qubits is not None else num_features
        self.seq_mode = seq_mode

        # Classical classifier backbone
        layers = []
        in_dim = num_features
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, num_features))
            layers.append(nn.ReLU())
            in_dim = num_features
        layers.append(nn.Linear(in_dim, 2))
        self.classical_classifier = nn.Sequential(*layers)

        # Quantum circuit placeholder
        self.quantum_circuit_builder = quantum_circuit_builder
        if mode == "quantum" and quantum_circuit_builder is None:
            raise ValueError("quantum_circuit_builder must be supplied for quantum mode")

        # Optional sequence tagging head
        if seq_mode:
            if vocab_size <= 0 or tagset_size <= 0:
                raise ValueError("vocab_size and tagset_size must be positive in seq_mode")
            self.word_embeddings = nn.Embedding(vocab_size, num_features)
            if n_qubits and n_qubits > 0:
                # Lazy import to avoid pulling in quantum libs unless needed
                from.qml_code import QLSTM as QuantumLSTM
                self.lstm = QuantumLSTM(num_features, seq_hidden_dim, n_qubits)
            else:
                self.lstm = nn.LSTM(num_features, seq_hidden_dim)
            self.hidden2tag = nn.Linear(seq_hidden_dim, tagset_size)

    def forward(self, x: torch.Tensor, seq: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        For classification: x -> [batch, features].
        For sequence tagging: seq -> [seq_len, batch] of token indices.
        """
        if self.seq_mode:
            if seq is None:
                raise ValueError("seq must be provided in sequence mode")
            embeds = self.word_embeddings(seq)
            lstm_out, _ = self.lstm(embeds)
            logits = self.hidden2tag(lstm_out)
            return F.log_softmax(logits, dim=-1)

        if self.mode == "classical":
            return self.classical_classifier(x)
        else:  # quantum
            # The quantum circuit is executed externally; here we return a placeholder.
            circuit, encoding, weights, observables = self.quantum_circuit_builder(self.n_qubits, self.depth)
            # In a real setup, the circuit would be run on a simulator and the
            # gradients propagated through the parameters. For this template we
            # simply return a random tensor matching the expected output shape.
            return torch.randn(x.shape[0], 2, device=x.device, dtype=x.dtype)

    def set_mode(self, mode: str):
        """Switch between 'classical' and 'quantum' mode."""
        if mode not in ("classical", "quantum"):
            raise ValueError("mode must be 'classical' or 'quantum'")
        self.mode = mode
