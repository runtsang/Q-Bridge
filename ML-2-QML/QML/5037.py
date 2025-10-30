"""Hybrid fraud detection model using Strawberry Fields photonics and TorchQuantum gates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple, List

import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate
from strawberryfields import operators as ops

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

# --- Parameters for photonic layers ------------------------------------
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

# --- Utility functions --------------------------------------------------
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
    """Create a Strawberry Fields program for the hybrid fraud detection model."""
    prog = sf.Program(2)
    with prog.context as q:
        _apply_layer(q, input_params, clip=False)
        for layer in layers:
            _apply_layer(q, layer, clip=True)
    return prog

# --- Quantum LSTM -------------------------------------------------------
class _QLSTMCell(tq.QuantumModule):
    """Gate‑based LSTM cell using TorchQuantum."""
    def __init__(self, n_wires: int):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
        )
        self.params = nn.ModuleList([tq.RX(has_params=True) for _ in range(n_wires)])
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
        self.encoder(qdev, x)
        for gate in self.params:
            gate(qdev)
        for i in range(self.n_wires - 1):
            tqf.cnot(qdev, wires=[i, i + 1])
        tqf.cnot(qdev, wires=[self.n_wires - 1, 0])
        return self.measure(qdev)

# --- FraudDetectionHybrid ------------------------------------------------
class FraudDetectionHybrid(nn.Module):
    """
    Hybrid fraud‑detection pipeline with a quantum encoder and a photonic head.
    """
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        *,
        n_qubits: int = 0,
        hidden_dim: int = 64,
        tagset_size: int = 5,
    ) -> None:
        super().__init__()
        self.use_quantum = n_qubits > 0
        self.head = build_fraud_detection_program(input_params, layers)
        if self.use_quantum:
            self.encoder = _QLSTMCell(n_qubits)
        else:
            self.encoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, tagset_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, feat)
        if isinstance(self.encoder, nn.LSTM):
            out, _ = self.encoder(x)
        else:
            out, _ = self.encoder(x)
        logits = self.output(out)
        return F.log_softmax(logits, dim=-1)

    # --- Quantum evaluation ------------------------------------------------
    def evaluate(
        self,
        shots: int = 1000,
        seed: int | None = None,
    ) -> List[float]:
        """
        Run the photonic circuit on a Strawberry Fields simulator and return
        the mean photon number of mode 0 as a simple observable.
        """
        engine = sf.Engine("tf", backend="default.qubit")
        if seed is not None:
            engine.set_random_seed(seed)
        result = engine.run(self.head, shots=shots)
        state = result.state
        photon_expect = state.expectation_value(ops.nmode(0))
        return [float(photon_expect)]

__all__ = ["FraudLayerParameters", "build_fraud_detection_program", "FraudDetectionHybrid"]
