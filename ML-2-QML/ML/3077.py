import torch
from torch import nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

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
# Classical fraud‑detection network
# --------------------------------------------------------------------------- #
def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    weight = torch.tensor([[params.bs_theta, params.bs_phi],
                           [params.squeeze_r[0], params.squeeze_r[1]]], dtype=torch.float32)
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

    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = linear
            self.activation = activation
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)

        def forward(self, inp: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            out = self.activation(self.linear(inp))
            out = out * self.scale + self.shift
            return out
    return Layer()

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    modules = [_layer_from_params(input_params, clip=False)]
    modules += [_layer_from_params(l, clip=True) for l in layers]
    modules.append(nn.Linear(2, 1))          # final read‑out
    return nn.Sequential(*modules)

# --------------------------------------------------------------------------- #
# Hybrid quantum‑LSTM cell
# --------------------------------------------------------------------------- #
class QLayer(tq.QuantumModule):
    """Small quantum circuit that implements a gate‑like transformation."""
    def __init__(self, n_wires: int):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
        )
        self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
        self.encoder(qdev, x)
        for w, gate in enumerate(self.params):
            gate(qdev, wires=w)
        for w in range(self.n_wires):
            tgt = 0 if w == self.n_wires - 1 else w + 1
            tqf.cnot(qdev, wires=[w, tgt])
        return self.measure(qdev)

class QLSTM(nn.Module):
    """LSTM cell where each gate is realised by a small quantum circuit."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        self.forget_map = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.input_map  = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.update_map = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.output_map = nn.Linear(input_dim + hidden_dim, n_qubits)

        self.forget_gate = QLayer(n_qubits)
        self.input_gate  = QLayer(n_qubits)
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
            f = torch.sigmoid(self.forget_gate(self.forget_map(combined)))
            i = torch.sigmoid(self.input_gate(self.input_map(combined)))
            g = torch.tanh(self.update_gate(self.update_map(combined)))
            o = torch.sigmoid(self.output_gate(self.output_map(combined)))
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
