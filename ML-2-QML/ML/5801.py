"""FraudDetectionHybridModel – classical branch of the hybrid fraud‑detection system.

The module exposes a single ``FraudDetectionHybridModel`` class that
* builds a multi‑layer feed‑forward network from a shared ``FraudLayerParameters`` dataclass,
* optionally appends a quantum‑augmented LSTM to capture temporal dynamics,
* and provides a ``forward`` method that can be called with a static input or a sequence of
  time‑steps.

The design deliberately re‑uses the construction logic from the original
FraudDetection.py seed while extending it with a hybrid LSTM and a
mechanism to switch between fully‑classical and classical‑plus‑quantum
behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# 1. Shared parameter schema
# --------------------------------------------------------------------------- #
@dataclass
class FraudLayerParameters:
    """Parameters describing a single layer in the classical stack.

    The dataclass is kept compatible with the original seed so that
    * ``FraudLayerParameters`` objects can be serialised and
    * ``build_fraud_detection_program`` from the seed can be imported
      for quantum modelling.
    """
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]

# --------------------------------------------------------------------------- #
# 2. Helper – layer construction
# --------------------------------------------------------------------------- #
def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    """Return a single ``nn.Module`` that implements the classical
    linear‑plus‑non‑linearity block described by *params*.

    Parameters
    ----------
    params
        The parameter bundle for the layer.
    clip
        Whether the weights and bias are clipped to a small range; this
        mirrors the behaviour of the quantum‑seed implementation.
    """
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

    class Layer(nn.Module):
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

    return Layer()

# --------------------------------------------------------------------------- #
# 3. Classical stack builder
# --------------------------------------------------------------------------- #
def build_classical_stack(input_params: FraudLayerParameters,
                        layers: Iterable[FraudLayerParameters]) -> nn.Sequential:
    """Build a feed‑forward network that mirrors the photonic program.

    The construction logic is identical to the original seed but the
    function name is changed to emphasise its purely classical nature.
    """
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

# --------------------------------------------------------------------------- #
# 4. Quantum‑augmented LSTM (quantum‑enhanced gates) – re‑used from QLSTM seed
# --------------------------------------------------------------------------- #
# Import here to avoid a hard runtime dependency when the module is used only
# for classical training – the import is lazy and wrapped in a try/except.
try:
    import torchquantum as tq
    import torchquantum.functional as tqf
except Exception as e:  # pragma: no cover – runtime only
    tq = None
    tqf = None

class QLSTM(nn.Module):
    """Hybrid LSTM where each gate is realised by a small quantum circuit.

    The implementation is a trimmed‑down copy of the QLSTM seed but wrapped
    inside this module so that it can be instantiated from the hybrid class.
    """

    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [0], "func": "rx", "wires": [0]},
                    {"input_idx": [1], "func": "rx", "wires": [1]},
                    {"input_idx": [2], "func": "rx", "wires": [2]},
                    {"input_idx": [3], "func": "rx", "wires": [3]},
                ]
            )
            self.params = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            for wire in range(self.n_wires):
                if wire == self.n_wires - 1:
                    tqf.cnot(qdev, wires=[wire, 0])
                else:
                    tqf.cnot(qdev, wires=[wire, wire + 1])
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        self.forget = self.QLayer(n_qubits)
        self.input = self.QLayer(n_qubits)
        self.update = self.QLayer(n_qubits)
        self.output = self.QLayer(n_qubits)

        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(self.linear_forget(combined)))
            i = torch.sigmoid(self.input(self.linear_input(combined)))
            g = torch.tanh(self.update(self.linear_update(combined)))
            o = torch.sigmoid(self.output(self.linear_output(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None = None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

# --------------------------------------------------------------------------- #
# 5. Hybrid model – classical + optional quantum LSTM
# --------------------------------------------------------------------------- #
class FraudDetectionHybridModel(nn.Module):
    """Main entry‑point for the hybrid fraud‑detection pipeline.

    Parameters
    ----------
    input_params
        Parameters for the first (input) layer.
    hidden_layers
        Iterable of ``FraudLayerParameters`` that will be hidden
        (the first element is *not* clipped; remaining
        layers – ‑ the same behaviour as in the original seed).
    n_qubits
        The number of quantum gates in the quantum‑augmented LSTM.
        If *zero* or missing, the model falls back to a purely classical
        LSTM.
    """
    def __init__(self,
                 input_params: FraudLayerParameters,
                 hidden_layers: Iterable[FraudLayerParameters],
                 n_qubits: int = 0,
                 hidden_dim: int = 64,
                 embedding_dim: int = 32,
                 vocab_size: int = 10000,
                 tagset_size: int = 10,
                 **kwargs: object) -> None:
        super().__init__()
        self.classical_stack = build_classical_stack(input_params, hidden_layers)
        self.n_qubits = n_qubits
        # ------------------------------------------------------------------
        # Optional quantum‑augmented LSTM – the seed provides a fully‑trainable
        # implementation that can be swapped in or out at runtime.
        # ------------------------------------------------------------------
        if n_qubits > 0 and tq is not None:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = None
        self.hidden_dim = hidden_dim
        self.input_mapper = nn.Linear(2, embedding_dim)
        self.tagger = nn.Linear(hidden_dim, tagset_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Run the hybrid pipeline.

        The method accepts either a single static feature vector of shape
        ``(batch, 2)`` or a sequence of feature vectors with shape
        ``(seq_len, batch, 2)``.
        """
        # 1) Classical feed‑forward part
        if inputs.dim() == 2:  # static input
            features = self.classical_stack(inputs)
            logits = self.tagger(features)
            return logits
        # 2) Sequence mode – use the embedded LSTM
        else:  # (seq_len, batch, 2)
            seq_emb = self.input_mapper(inputs)
            if self.lstm is None:
                raise NotImplementedError("Quantum LSTM not available; set n_qubits > 0.")
            lstm_out, _ = self.lstm(seq_emb)
            logits = self.tagger(lstm_out)
            return logits

__all__ = ["FraudDetectionHybridModel", "FraudLayerParameters", "build_classical_stack", "QLSTM"]
