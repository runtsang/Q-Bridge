"""Quantum‑enhanced fraud detection circuit with variational LSTM integration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple, List

import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Rgate, Sgate, Kgate

import pennylane as qml
import torch
from torch import nn

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

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _apply_layer(modes: Sequence, params: FraudLayerParameters, *, clip: bool) -> None:
    BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
    for i, phase in enumerate(params.phases):
        Rgate(phase) | modes[i]
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        Sgate(r if not clip else _clip(r, 5), phi) | modes[i]
    BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
    for i, phase in enumerate(params.phases):
        Rgate(phase) | modes[i]
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        Dgate(r if not clip else _clip(r, 5), phi) | modes[i]
    for i, k in enumerate(params.kerr):
        Kgate(k if not clip else _clip(k, 1)) | modes[i]

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> sf.Program:
    program = sf.Program(2)
    with program.context as q:
        _apply_layer(q, input_params, clip=False)
        for layer in layers:
            _apply_layer(q, layer, clip=True)
    return program

class QuantumLSTM(nn.Module):
    """Quantum LSTM implemented with PennyLane."""
    class QGate(nn.Module):
        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.device = qml.device("default.qubit", wires=n_wires)
            @qml.qnode(self.device, interface="torch")
            def circuit(x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
                for w in range(self.n_wires):
                    qml.RX(x[w], wires=w)
                for w in range(self.n_wires):
                    qml.RX(params[w], wires=w)
                for w in range(self.n_wires - 1):
                    qml.CNOT(wires=[w, w + 1])
                qml.CNOT(wires=[self.n_wires - 1, 0])
                return qml.expval(qml.PauliZ(0))
            self.circuit = circuit
            self.params = nn.Parameter(torch.randn(n_wires))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.circuit(x, self.params)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        self.forget = self.QGate(n_qubits)
        self.input = self.QGate(n_qubits)
        self.update = self.QGate(n_qubits)
        self.output = self.QGate(n_qubits)

        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

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

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs: List[torch.Tensor] = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(self.linear_forget(combined)))
            i = torch.sigmoid(self.input(self.linear_input(combined)))
            g = torch.tanh(self.update(self.linear_update(combined)))
            o = torch.sigmoid(self.output(self.linear_output(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        return torch.cat(outputs, dim=0), (hx, cx)

class FraudDetectionHybrid(nn.Module):
    """Hybrid model that first processes inputs with a photonic circuit
    (Strawberry Fields) and then models temporal dependencies with a
    PennyLane quantum LSTM."""
    def __init__(
        self,
        extractor_params: FraudLayerParameters,
        extractor_layers: Iterable[FraudLayerParameters],
        hidden_dim: int = 32,
        n_qubits: int = 4,
    ) -> None:
        super().__init__()
        self.extractor_program = build_fraud_detection_program(extractor_params, extractor_layers)
        self.lstm = QuantumLSTM(2, hidden_dim, n_qubits)
        self.classifier = nn.Linear(hidden_dim, 1)

    def _photonic_forward(self, vec: torch.Tensor) -> torch.Tensor:
        # Placeholder: return the raw vector; in practice run the SF program.
        return vec

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        # seq shape: (seq_len, batch, 2)
        photonic_features = torch.stack([self._photonic_forward(v) for v in seq.unbind(dim=0)], dim=0)
        lstm_out, _ = self.lstm(photonic_features)
        logits = self.classifier(lstm_out[:, -1, :])  # last hidden state
        return torch.sigmoid(logits).squeeze(-1)

__all__ = ["FraudLayerParameters", "build_fraud_detection_program", "QuantumLSTM", "FraudDetectionHybrid"]
