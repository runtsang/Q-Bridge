import pennylane as qml
import pennylane.numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Iterable, Tuple

# --------------------------------------------------------------------------- #
# Shared data‑structures
# --------------------------------------------------------------------------- #
@dataclass
class FraudLayerParameters:
    """Parameters describing one photonic layer, also usable for the classical twin."""
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]

# --------------------------------------------------------------------------- #
# Quantum fraud‑detection circuit
# --------------------------------------------------------------------------- #
def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _apply_layer(params: FraudLayerParameters, clip: bool) -> None:
    qml.RX(params.bs_theta, wires=0)
    qml.RX(params.bs_phi, wires=1)
    qml.RZ(params.phases[0], wires=0)
    qml.RZ(params.phases[1], wires=1)
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        r_val = _clip(r, 5) if clip else r
        phi_val = _clip(phi, 5) if clip else phi
        qml.RX(r_val, wires=i)
        qml.RZ(phi_val, wires=i)
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        r_val = _clip(r, 5) if clip else r
        phi_val = _clip(phi, 5) if clip else phi
        qml.RX(r_val, wires=i)
        qml.RZ(phi_val, wires=i)
    for i, k in enumerate(params.kerr):
        k_val = _clip(k, 1) if clip else k
        qml.RZ(k_val, wires=i)
    # Entangling gate to mimic beam‑splitter
    qml.CNOT(0, 1)

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters]
) -> qml.QNode:
    all_params = [input_params] + list(layers)
    dev = qml.device("default.qubit", wires=2)
    @qml.qnode(dev)
    def circuit(x):
        qml.RX(x[0], wires=0)
        qml.RX(x[1], wires=1)
        for i, params in enumerate(all_params):
            _apply_layer(params, clip=(i!= 0))
        return qml.expval(qml.PauliZ(0))
    return circuit

# --------------------------------------------------------------------------- #
# Quantum layer used inside the quantum LSTM
# --------------------------------------------------------------------------- #
class QLayer:
    """Small quantum circuit that implements a gate‑like transformation."""
    def __init__(self, n_wires: int):
        self.n_wires = n_wires
        self.params = np.random.randn(n_wires)
        self.device = qml.device("default.qubit", wires=n_wires)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        outputs = []
        for i in range(batch_size):
            @qml.qnode(self.device)
            def circuit(x_i):
                for j in range(self.n_wires):
                    qml.RX(x_i[j], wires=j)
                    qml.RX(self.params[j], wires=j)
                for j in range(self.n_wires - 1):
                    qml.CNOT(j, j + 1)
                return [qml.expval(qml.PauliZ(j)) for j in range(self.n_wires)]
            outputs.append(circuit(x[i].cpu().numpy()))
        return torch.tensor(outputs, dtype=torch.float32, device=x.device)

# --------------------------------------------------------------------------- #
# Quantum‑augmented LSTM
# --------------------------------------------------------------------------- #
class QLSTM(nn.Module):
    """LSTM cell where each gate is realised by a small quantum circuit."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        self.forget_linear = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.input_linear = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.update_linear = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.output_linear = nn.Linear(input_dim + hidden_dim, n_qubits)

        self.forget_gate = QLayer(n_qubits)
        self.input_gate = QLayer(n_qubits)
        self.update_gate = QLayer(n_qubits)
        self.output_gate = QLayer(n_qubits)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_gate(self.forget_linear(combined)))
            i = torch.sigmoid(self.input_gate(self.input_linear(combined)))
            g = torch.tanh(self.update_gate(self.update_linear(combined)))
            o = torch.sigmoid(self.output_gate(self.output_linear(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        return torch.cat(outputs, dim=0), (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

# --------------------------------------------------------------------------- #
# Composite tagger that can switch between classical and quantum LSTM
# --------------------------------------------------------------------------- #
class LSTMTagger(nn.Module):
    """Sequence tagging model that uses the fraud‑detector as a feature extractor."""
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        fraud_layers: Iterable[FraudLayerParameters] | None = None,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.fraud_extractor = None
        if fraud_layers is not None:
            fraud_list = list(fraud_layers)
            self.fraud_extractor = build_fraud_detection_program(
                input_params=fraud_list[0],
                layers=fraud_list[1:],
            )
        lstm_input_dim = embedding_dim + 1 if self.fraud_extractor is not None else embedding_dim
        if n_qubits > 0:
            self.lstm = QLSTM(lstm_input_dim, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = nn.LSTM(lstm_input_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)  # shape: (seq_len, embedding_dim)
        if self.fraud_extractor is not None:
            fraud_feats = []
            for i in range(embeds.size(0)):
                vec = embeds[i, :2].detach().cpu().numpy()
                out = self.fraud_extractor(vec)
                fraud_feats.append([float(out)])
            fraud_feats = torch.tensor(fraud_feats, dtype=torch.float32, device=embeds.device)
            lstm_inputs = torch.cat([embeds, fraud_feats], dim=1)
        else:
            lstm_inputs = embeds
        lstm_out, _ = self.lstm(lstm_inputs.unsqueeze(1))
        tag_logits = self.hidden2tag(lstm_out.squeeze(1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "QLSTM",
    "LSTMTagger",
]
