import pennylane as qml
import torch
import torch.nn as nn
import networkx as nx
import itertools
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple, List

@dataclass
class LayerParams:
    """Quantum‑style parameters that mimic the photonic layer."""
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]

def _clip(v: float, bound: float) -> float:
    return max(-bound, min(bound, v))

def _apply_layer(qc: qml.QuantumCircuit,
                 params: LayerParams,
                 clip: bool,
                 wire: int = 0):
    """Encode a single transaction vector into a small qubit register."""
    # Beam splitter‑style rotations
    qml.RX(params.bs_theta, wires=wire)
    qml.RY(params.bs_phi, wires=wire)
    for i, phase in enumerate(params.phases):
        qml.RZ(phase, wires=i)
    # Squeezing and displacement simulated with rotations
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        qml.S(r if not clip else _clip(r, 5), wires=i)
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        qml.RX(r if not clip else _clip(r, 5), wires=i)
    # Kerr‑like non‑linearity with RZ
    for i, k in enumerate(params.kerr):
        qml.RZ(k if not clip else _clip(k, 1), wires=i)

def build_fraud_detection_circuit(input_params: LayerParams,
                                  layers: Iterable[LayerParams],
                                  wires: int = 2) -> qml.QuantumCircuit:
    """Return a PennyLane circuit that mirrors the photonic fraud‑detection stack."""
    qc = qml.QuantumCircuit(wires=wires)
    _apply_layer(qc, input_params, clip=False)
    for l in layers:
        _apply_layer(qc, l, clip=True)
    return qc

# ------------------------------------------------------------------
# Quantum LSTM block (PennyLane)
# ------------------------------------------------------------------
class QuantumLSTMBlock(nn.Module):
    def __init__(self, n_qubits: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.dev = qml.device("default.qubit", wires=n_qubits)
        @qml.qnode(self.dev, interface="torch")
        def circuit(x):
            # Encode input
            for i in range(self.n_qubits):
                qml.RX(x[i], wires=i)
            # Entangle
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            qml.CNOT(wires=[self.n_qubits - 1, 0])
            # Return expectation of PauliZ on first qubit
            return qml.expval(qml.PauliZ(0))
        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, n_qubits)
        return self.circuit(x)

# ------------------------------------------------------------------
# Quantum‑enhanced fraud detection model
# ------------------------------------------------------------------
class FraudDetectionHybrid(nn.Module):
    """Quantum‑enhanced fraud‑detection model that fuses a quantum LSTM
    with a classical LSTM for sequential classification."""
    def __init__(self,
                 input_dim: int = 2,
                 hidden_dim: int = 8,
                 n_qubits: int = 4):
        super().__init__()
        self.quantum_lstm = QuantumLSTMBlock(n_qubits)
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, 2)
        batch, seq_len, _ = x.shape
        qs = []
        for t in range(seq_len):
            qt = self.quantum_lstm(x[:, t, :])
            qs.append(qt.unsqueeze(1))
        qs = torch.cat(qs, dim=1)  # (batch, seq_len, 1)
        lstm_out, _ = self.lstm(qs)
        out = self.fc(lstm_out[:, -1, :])
        return torch.sigmoid(out)

__all__ = ["LayerParams",
           "build_fraud_detection_circuit",
           "QuantumLSTMBlock",
           "FraudDetectionHybrid"]
