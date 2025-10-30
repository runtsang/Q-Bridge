"""HybridQuantumClassifier: image and sequence classifier with quantum back‑end.

The class can operate in two distinct modes:
    - 'image': a CNN backbone followed by a quantum expectation head.
    -'seq': an embedding layer + optional quantum LSTM cell for tagging.

Quantum components are defined in quantum_utils.py and are imported lazily.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import quantum utilities lazily to keep the module usable without Qiskit.
try:
    from quantum_utils import QuantumHybridLayer, QuantumLSTMCell
except Exception:  # pragma: no cover
    QuantumHybridLayer = None  # type: ignore
    QuantumLSTMCell = None  # type: ignore


class HybridQuantumClassifier(nn.Module):
    """Unified classifier for image and sequence tasks."""
    def __init__(
        self,
        mode: str = "image",
        *,
        in_channels: int = 3,
        num_classes: int = 2,
        vocab_size: int = 10000,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        n_qubits: int = 0,
        shots: int = 1024,
        **kwargs,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.n_qubits = n_qubits
        self.shots = shots

        if mode == "image":
            self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5, stride=2, padding=1)
            self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
            self.drop1 = nn.Dropout2d(p=0.2)
            self.drop2 = nn.Dropout2d(p=0.5)

            # Flattened feature size after the conv layers (for 32×32 inputs)
            flat_dim = 15 * 6 * 6  # 540

            self.fc1 = nn.Linear(flat_dim, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 1)

            if n_qubits > 0 and QuantumHybridLayer is not None:
                self.quantum_head = QuantumHybridLayer(1, n_qubits=n_qubits, shots=shots, **kwargs)
            else:
                self.quantum_head = nn.Sigmoid()

        elif mode == "seq":
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.hidden_dim = hidden_dim

            if n_qubits > 0 and QuantumLSTMCell is not None:
                self.lstm = QuantumLSTMCell(embedding_dim, hidden_dim, n_qubits=n_qubits, shots=shots, **kwargs)
            else:
                self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

            self.hidden2tag = nn.Linear(hidden_dim, num_classes)

        else:
            raise ValueError("mode must be either 'image' or'seq'")

    def _conv_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def forward(self, x: torch.Tensor):
        if self.mode == "image":
            x = self._conv_forward(x)
            probs = self.quantum_head(x)
            return torch.cat((probs, 1 - probs), dim=-1)

        else:  # seq
            emb = self.embedding(x)  # shape: (batch, seq_len, embed_dim)
            if isinstance(self.lstm, nn.LSTM):
                lstm_out, _ = self.lstm(emb)  # shape: (batch, seq_len, hidden_dim)
            else:
                # QuantumLSTMCell processing
                batch, seq_len, _ = emb.shape
                hx = torch.zeros(batch, self.hidden_dim, device=emb.device)
                cx = torch.zeros(batch, self.hidden_dim, device=emb.device)
                outputs = []
                for t in range(seq_len):
                    hx, (hx, cx) = self.lstm(emb[:, t, :], (hx, cx))
                    outputs.append(hx.unsqueeze(1))
                lstm_out = torch.cat(outputs, dim=1)
            logits = self.hidden2tag(lstm_out)
            return F.log_softmax(logits, dim=-1)


__all__ = ["HybridQuantumClassifier"]
