import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple, List, Optional

@dataclass
class FraudLayerParameters:
    """Parameters describing a fully connected layer in the classical model."""
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _create_fraud_layer(params: FraudLayerParameters, clip: bool = False) -> nn.Module:
    weight = torch.tensor(
        [
            [params.bs_theta, params.bs_phi],
            [params.squeeze_r[0], params.squeeze_r[1]],
        ],
        dtype=torch.float32,
    )
    bias = torch.tensor(params.phases, dtype=torch.float32)
    if clip:
        weight = weight.clamp(-5.0, 5.0)
        bias = bias.clamp(-5.0, 5.0)
    linear = nn.Linear(2, 2)
    with torch.no_grad():
        linear.weight.copy_(weight)
        linear.bias.copy_(bias)
    activation = nn.Tanh()
    scale = torch.tensor(params.displacement_r, dtype=torch.float32)
    shift = torch.tensor(params.displacement_phi, dtype=torch.float32)

    class FraudLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = linear
            self.activation = activation
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            outputs = self.activation(self.linear(inputs))
            outputs = outputs * self.scale + self.shift
            return outputs

    return FraudLayer()

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Sequence[FraudLayerParameters],
) -> nn.Sequential:
    modules: List[nn.Module] = [_create_fraud_layer(input_params, clip=False)]
    modules.extend(_create_fraud_layer(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

class FullyConnectedLayer(nn.Module):
    """A lightweight classical approximation of a quantum fully‑connected layer."""
    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        expectation = torch.tanh(self.linear(values)).mean(dim=0)
        return expectation.detach().numpy()

class QCNNModel(nn.Module):
    """A convolution‑inspired classical network mirroring the quantum QCNN design."""
    def __init__(self) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

class QLSTMGen236(nn.Module):
    """
    Hybrid tagger that can operate in pure classical mode or with quantum back‑ends.
    Parameters
    ----------
    embedding_dim : int
        Dimensionality of the word embeddings.
    hidden_dim : int
        Hidden state dimensionality of the LSTM.
    vocab_size : int
        Size of the vocabulary.
    tagset_size : int
        Number of possible tags.
    n_qubits : int, optional
        If > 0, the LSTM gates are implemented with quantum circuits.
    fraud_params : Optional[Sequence[FraudLayerParameters]], optional
        Parameters for a fraud‑detection sub‑module applied to the embeddings.
    use_qcnn : bool, optional
        If True, the final classification head is a QCNN‑style network.
    use_fcl : bool, optional
        If True, the final classification head is a fully‑connected layer.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        fraud_params: Optional[Sequence[FraudLayerParameters]] = None,
        use_qcnn: bool = False,
        use_fcl: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Classical LSTM is used in the ML module; quantum LSTM can be swapped in the QML module.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

        self.fraud_module: Optional[nn.Sequential] = None
        if fraud_params:
            self.fraud_module = build_fraud_detection_program(
                fraud_params[0], fraud_params[1:]
            )

        self.qcnn_head: Optional[QCNNModel] = None
        if use_qcnn:
            self.qcnn_head = QCNNModel()

        self.fcl_head: Optional[FullyConnectedLayer] = None
        if use_fcl:
            self.fcl_head = FullyConnectedLayer()

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the hybrid tagger.
        Parameters
        ----------
        sentence : torch.Tensor
            Tensor of word indices, shape (seq_len, batch).
        Returns
        -------
        torch.Tensor
            Log‑softmax of tag logits, shape (seq_len, tagset_size).
        """
        embeds = self.word_embeddings(sentence)
        if self.fraud_module:
            embeds = self.fraud_module(embeds)

        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))

        if self.qcnn_head:
            tag_logits = self.qcnn_head(lstm_out.squeeze(0))
        elif self.fcl_head:
            # The FCL expects a flat iterable of values.
            tag_logits = torch.tensor(
                self.fcl_head.run(lstm_out.squeeze(0).detach().numpy()),
                dtype=torch.float32,
            )
        else:
            tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))

        return F.log_softmax(tag_logits, dim=1)

__all__ = ["QLSTMGen236"]
