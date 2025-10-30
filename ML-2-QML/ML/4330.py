"""Classical hybrid binary classifier with optional auto‑encoder and quantum‑inspired head.

This module combines a CNN feature extractor, a configurable feed‑forward
classifier (build_classifier_circuit), and an optional auto‑encoder.
The “quantum head” is a lightweight sigmoid layer that mimics the
expectation value of a two‑qubit circuit, enabling cross‑framework
experimentation without pulling in Qiskit.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Iterable

# --------------------------------------------------------------------------- #
#  Classical classifier factory (inspired by QuantumClassifierModel.py)       #
# --------------------------------------------------------------------------- #
def build_classifier_circuit(num_features: int, depth: int = 2) -> nn.Sequential:
    """
    Construct a feed‑forward classifier with a depth‑controlled number of
    hidden layers.  The interface mirrors the quantum helper from the seed
    but uses only PyTorch primitives.
    """
    layers: list[nn.Module] = []
    in_dim = num_features
    for _ in range(depth):
        layers.append(nn.Linear(in_dim, num_features))
        layers.append(nn.ReLU(inplace=True))
        in_dim = num_features
    layers.append(nn.Linear(in_dim, 2))
    return nn.Sequential(*layers)

# --------------------------------------------------------------------------- #
#  Auto‑encoder (inspired by Autoencoder.py)                                 #
# --------------------------------------------------------------------------- #
class AutoencoderConfig:
    def __init__(self, input_dim: int, latent_dim: int = 32,
                 hidden_dims: Tuple[int, int] = (128, 64),
                 dropout: float = 0.1):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout

class AutoencoderNet(nn.Module):
    def __init__(self, config: AutoencoderConfig):
        super().__init__()
        encoder_layers = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU(inplace=True))
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU(inplace=True))
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

def Autoencoder(input_dim: int, *, latent_dim: int = 32,
                hidden_dims: Tuple[int, int] = (128, 64),
                dropout: float = 0.1) -> AutoencoderNet:
    cfg = AutoencoderConfig(input_dim, latent_dim, hidden_dims, dropout)
    return AutoencoderNet(cfg)

# --------------------------------------------------------------------------- #
#  Quantum‑inspired head (HybridFunction)                                    #
# --------------------------------------------------------------------------- #
class HybridFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float = 0.0) -> torch.Tensor:
        out = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        (out,) = ctx.saved_tensors
        return grad_output * out * (1 - out), None

# --------------------------------------------------------------------------- #
#  Hybrid binary classifier                                                 #
# --------------------------------------------------------------------------- #
class HybridBinaryClassifier(nn.Module):
    """
    CNN feature extractor → optional auto‑encoder → feed‑forward classifier.
    The final head can be a classical linear layer or a quantum‑inspired
    sigmoid that mimics the expectation value of a two‑qubit circuit.
    """
    def __init__(self,
                 use_autoencoder: bool = False,
                 autoencoder_cfg: dict | None = None,
                 use_quantum_head: bool = False,
                 classifier_depth: int = 2):
        super().__init__()
        # CNN backbone
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        self.use_autoencoder = use_autoencoder
        if use_autoencoder:
            if autoencoder_cfg is None:
                autoencoder_cfg = {}
            self.autoencoder = Autoencoder(**autoencoder_cfg)

        self.use_quantum_head = use_quantum_head
        if not use_quantum_head:
            # classical head: a tiny feed‑forward network
            self.classifier = build_classifier_circuit(1, depth=classifier_depth)
        else:
            self.shift = 0.0  # shift for the sigmoid approximation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN feature extraction
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)

        # optional auto‑encoder compression
        if self.use_autoencoder:
            x = self.autoencoder.encode(x)

        # fully‑connected layers
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # final head
        if self.use_quantum_head:
            probs = HybridFunction.apply(x.squeeze(), self.shift)
        else:
            probs = self.classifier(x.squeeze())

        # return a 2‑column probability tensor
        return torch.cat((probs, 1 - probs), dim=-1)

# --------------------------------------------------------------------------- #
#  Classical LSTM tagger (inspired by QLSTM.py)                              #
# --------------------------------------------------------------------------- #
class QLSTM(nn.Module):
    """Drop‑in replacement using classical linear gates."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.forget_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(self, inputs: torch.Tensor,
                states: Tuple[torch.Tensor, torch.Tensor] | None = None
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
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

    def _init_states(self, inputs: torch.Tensor,
                     states: Tuple[torch.Tensor, torch.Tensor] | None = None
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

class LSTMTagger(nn.Module):
    """Sequence tagging model that uses either :class:`QLSTM` or ``nn.LSTM``."""
    def __init__(self, embedding_dim: int, hidden_dim: int,
                 vocab_size: int, tagset_size: int,
                 n_qubits: int = 0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = [
    "HybridBinaryClassifier",
    "HybridFunction",
    "Autoencoder",
    "AutoencoderNet",
    "AutoencoderConfig",
    "build_classifier_circuit",
    "QLSTM",
    "LSTMTagger",
]
