import torch
import torch.nn as nn
import torch.nn.functional as F
import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate, N
import torchquantum as tq
import torchquantum.functional as tqf
from dataclasses import dataclass
from typing import Iterable, Tuple, List, Optional

@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]

def _clip(val: float, bound: float) -> float:
    return max(-bound, min(bound, val))

def _build_quantum_photonic_layer(params: FraudLayerParameters, *, clip: bool = False) -> sf.Program:
    prog = sf.Program(2)
    with prog.context as q:
        BSgate(params.bs_theta, params.bs_phi) | (q[0], q[1])
        for i, phase in enumerate(params.phases):
            Rgate(phase) | q[i]
        for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
            Sgate(r if not clip else _clip(r, 5), phi) | q[i]
        BSgate(params.bs_theta, params.bs_phi) | (q[0], q[1])
        for i, phase in enumerate(params.phases):
            Rgate(phase) | q[i]
        for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
            Dgate(r if not clip else _clip(r, 5), phi) | q[i]
        for i, k in enumerate(params.kerr):
            Kgate(k if not clip else _clip(k, 1)) | q[i]
    return prog

class QuantumPhotonicFeatureExtractor(nn.Module):
    """Extracts 2‑dimensional features using a small photonic circuit per timestep."""
    def __init__(self, input_params: FraudLayerParameters, layers: List[FraudLayerParameters]):
        super().__init__()
        self.progs = [_build_quantum_photonic_layer(input_params, clip=False)]
        self.progs.extend(_build_quantum_photonic_layer(l, clip=True) for l in layers)
        self.backend = sf.Engine("fock", backend_options={"cutoff_dim": 5})

    def forward(self, x: torch.Tensor):
        # x shape: (batch*seq_len, 2)
        batch_seq_len, _ = x.shape
        outputs = []
        for i in range(batch_seq_len):
            prog = self.progs[i % len(self.progs)]
            result = self.backend.run(prog, args=[x[i]])
            n0 = result.expectation_value(N, q=0)
            n1 = result.expectation_value(N, q=1)
            outputs.append(torch.tensor([n0, n1], device=x.device))
        return torch.stack(outputs)

class QLSTM(nn.Module):
    """Quantum‑enhanced LSTM cell that maps classical vectors to quantum‑encoded states."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [0], "func": "rx", "wires": [0]},
                    {"input_idx": [1], "func": "rx", "wires": [1]},
                ]
            )
            self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
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

    def forward(self, inputs: torch.Tensor, states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
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

    def _init_states(self, inputs: torch.Tensor, states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))

class FraudDetectionHybrid(nn.Module):
    """Full quantum‑enhanced fraud detection model."""
    def __init__(self,
                 input_params: FraudLayerParameters,
                 layers: List[FraudLayerParameters],
                 hidden_dim: int = 32,
                 n_qubits: int = 4):
        super().__init__()
        self.extractor = QuantumPhotonicFeatureExtractor(input_params, layers)
        self.seq_model = QLSTM(2, hidden_dim, n_qubits)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor):
        # x shape: (batch, seq_len, 2)
        batch, seq_len, _ = x.shape
        x_flat = x.reshape(batch * seq_len, 2)
        features = self.extractor(x_flat)
        features = features.reshape(batch, seq_len, -1)
        features = features.permute(1, 0, 2)
        outputs, _ = self.seq_model(features)
        logits = self.classifier(outputs[-1])
        return logits

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
    hidden_dim: int = 32,
    n_qubits: int = 4,
) -> FraudDetectionHybrid:
    """Factory that returns a FraudDetectionHybrid instance with quantum processing."""
    return FraudDetectionHybrid(input_params, list(layers), hidden_dim, n_qubits)

__all__ = ["FraudLayerParameters", "FraudDetectionHybrid", "build_fraud_detection_program"]
