import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
#  Classical components – CNN and (optional) LSTM tagger
# --------------------------------------------------------------------------- #
class ClassicalCNN(nn.Module):
    """Simple 2‑layer CNN producing a fixed‑size feature vector."""
    def __init__(self, out_features: int = 120) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, out_features)
        self.out_features = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        return self.fc3(x)


class ClassicalLSTMTagger(nn.Module):
    """Classic LSTM tagger used when quantum gates are disabled."""
    def __init__(self, input_dim: int, hidden_dim: int, tagset_size: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        # seq: (seq_len, batch, input_dim)
        lstm_out, _ = self.lstm(seq)
        return F.log_softmax(self.hidden2tag(lstm_out), dim=-1)


# --------------------------------------------------------------------------- #
#  Quantum‑enabled head (imported from the quantum module)
# --------------------------------------------------------------------------- #
# The quantum classes are defined in the accompanying qml module.
# Import them lazily to keep the classical module lightweight.
try:
    from.qml_module import QuantumHybridHead  # type: ignore
except Exception:  # pragma: no cover
    # Fallback stub – will be replaced by the real quantum implementation
    class QuantumHybridHead(nn.Module):
        def __init__(self, *_, **__):
            super().__init__()
            raise RuntimeError("Quantum libraries not available in this environment.")

# --------------------------------------------------------------------------- #
#  Combined hybrid classifier
# --------------------------------------------------------------------------- #
class HybridBinaryClassifier(nn.Module):
    """
    End‑to‑end binary classifier that optionally uses a quantum expectation
    layer as the final decision head.  The model can also process
    sequential data through a classical LSTM tagger before the quantum head.
    """
    def __init__(
        self,
        use_lstm: bool = False,
        lstm_hidden: int = 128,
        lstm_tagset: int = 2,
        n_qubits: int = 2,
        backend=None,
        shots: int = 100,
        shift: float = np.pi / 2,
    ) -> None:
        super().__init__()
        self.cnn = ClassicalCNN()
        self.use_lstm = use_lstm
        if use_lstm:
            self.lstm = ClassicalLSTMTagger(
                input_dim=self.cnn.out_features,
                hidden_dim=lstm_hidden,
                tagset_size=lstm_tagset,
            )
        else:
            self.lstm = None
        # Quantum head – a parameterised expectation layer
        self.quantum_head = QuantumHybridHead(
            n_qubits=n_qubits, backend=backend, shots=shots, shift=shift
        )
        # Final linear layer that maps the quantum expectation to a logit
        self.classifier = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 3, H, W)
        features = self.cnn(x)
        if self.use_lstm:
            # Convert to (seq_len=1, batch, features) for LSTM
            seq = features.unsqueeze(0)
            lstm_out = self.lstm(seq)
            # lstm_out: (seq_len, batch, tagset)
            # Use the tagset dimension (class scores) as the new feature
            features = lstm_out.squeeze(0)
        # Quantum head expects a 1‑D vector per sample
        quantum_out = self.quantum_head(features)
        logits = self.classifier(quantum_out)
        probs = torch.sigmoid(logits)
        return torch.cat((probs, 1 - probs), dim=-1)


__all__ = [
    "ClassicalCNN",
    "ClassicalLSTMTagger",
    "HybridBinaryClassifier",
]
