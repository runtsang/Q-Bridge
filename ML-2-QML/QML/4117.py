"""Quantum‑enhanced LSTM tagger that replaces classical gates with TorchQuantum circuits
and embeds a quantum self‑attention block.  The API mirrors the classical HybridQLSTM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchquantum as tq
import torchquantum.functional as tqf


# --------------------------------------------------------------------------- #
#   Fraud‑Detection style parameter block (identical to the classical version)
# --------------------------------------------------------------------------- #
@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> tq.QuantumModule:
    """Build a quantum‑style linear block that mimics the photonic layer."""
    class FraudQuantumLayer(tq.QuantumModule):
        def __init__(self) -> None:
            super().__init__()
            # Parameterised gates that match the photonic circuit
            self.bs = tq.BSgate()
            self.r = [tq.Rgate() for _ in range(2)]
            self.s = [tq.Sgate(has_params=True) for _ in range(2)]
            self.d = [tq.Dgate(has_params=True) for _ in range(2)]
            self.k = [tq.Kgate(has_params=True) for _ in range(2)]
            self.measure = tq.MeasureAll(tq.PauliZ)

            # Store parameters as tensors for easy update
            self.register_buffer("bs_theta", torch.tensor(params.bs_theta))
            self.register_buffer("bs_phi", torch.tensor(params.bs_phi))
            self.register_buffer("phases", torch.tensor(params.phases))
            self.register_buffer("squeeze_r", torch.tensor(params.squeeze_r))
            self.register_buffer("squeeze_phi", torch.tensor(params.squeeze_phi))
            self.register_buffer("displacement_r", torch.tensor(params.displacement_r))
            self.register_buffer("displacement_phi", torch.tensor(params.displacement_phi))
            self.register_buffer("kerr", torch.tensor(params.kerr))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            bsz = x.shape[0]
            dev = tq.QuantumDevice(2, bsz, device=x.device)

            # Apply BS gate
            self.bs(dev, theta=self.bs_theta, phi=self.bs_phi)

            # Apply phase gates
            for i, phase in enumerate(self.phases):
                self.r[i](dev, theta=phase)

            # Apply squeezing gates
            for i, (r, phi) in enumerate(zip(self.squeeze_r, self.squeeze_phi)):
                self.s[i](dev, r=r if not clip else _clip(r, 5), theta=phi)

            # Apply second BS gate
            self.bs(dev, theta=self.bs_theta, phi=self.bs_phi)

            # Apply phase gates again
            for i, phase in enumerate(self.phases):
                self.r[i](dev, theta=phase)

            # Apply displacement gates
            for i, (r, phi) in enumerate(zip(self.displacement_r, self.displacement_phi)):
                self.d[i](dev, r=r if not clip else _clip(r, 5), theta=phi)

            # Apply Kerr gates
            for i, k in enumerate(self.kerr):
                self.k[i](dev, k=k if not clip else _clip(k, 1))

            return self.measure(dev)

    return FraudQuantumLayer()


# --------------------------------------------------------------------------- #
#   Quantum Self‑Attention block
# --------------------------------------------------------------------------- #
class QuantumSelfAttention(tq.QuantumModule):
    """A small qiskit‑style self‑attention circuit built with TorchQuantum."""

    def __init__(self, n_qubits: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.r = [tq.Rgate(has_params=True) for _ in range(n_qubits)]
        self.crx = [tq.CRXgate(has_params=True) for _ in range(n_qubits - 1)]
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        dev = tq.QuantumDevice(self.n_qubits, bsz, device=x.device)

        # Rotation layer
        for i in range(self.n_qubits):
            self.r[i](dev, theta=x[:, i])

        # Entangling layer
        for i in range(self.n_qubits - 1):
            self.crx[i](dev, theta=x[:, i])

        return self.measure(dev)


# --------------------------------------------------------------------------- #
#   Quantum‑enhanced LSTM cell
# --------------------------------------------------------------------------- #
class QuantumLSTMCell(tq.QuantumModule):
    """LSTM cell where each gate is implemented by a quantum circuit."""

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Linear projections to quantum space
        self.lin_f = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.lin_i = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.lin_g = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.lin_o = nn.Linear(input_dim + hidden_dim, n_qubits)

        # Quantum gates for each LSTM gate
        self.qgate_f = QuantumSelfAttention(n_qubits)
        self.qgate_i = QuantumSelfAttention(n_qubits)
        self.qgate_g = QuantumSelfAttention(n_qubits)
        self.qgate_o = QuantumSelfAttention(n_qubits)

    def forward(self, x: torch.Tensor, hx: torch.Tensor, cx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([x, hx], dim=1)
        f = torch.sigmoid(self.qgate_f(self.lin_f(combined)))
        i = torch.sigmoid(self.qgate_i(self.lin_i(combined)))
        g = torch.tanh(self.qgate_g(self.lin_g(combined)))
        o = torch.sigmoid(self.qgate_o(self.lin_o(combined)))

        cx = f * cx + i * g
        hx = o * torch.tanh(cx)
        return hx, cx


# --------------------------------------------------------------------------- #
#   Hybrid LSTM Tagger (quantum)
# --------------------------------------------------------------------------- #
class HybridQLSTM(tq.QuantumModule):
    """Quantum‑enhanced tagging model that mirrors the classical API."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        use_qattn: bool = False,
        fraud_params: Iterable[FraudLayerParameters] | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Fraud‑detection style pre‑processing (quantum version)
        if fraud_params is not None:
            layers = [_layer_from_params(next(fraud_params), clip=False)]
            layers.extend(_layer_from_params(p, clip=True) for p in fraud_params)
            self.fraud_block = nn.Sequential(*layers)
        else:
            self.fraud_block = nn.Identity()

        # Quantum self‑attention pre‑processor
        self.use_qattn = use_qattn
        if use_qattn:
            self.attn = QuantumSelfAttention(embedding_dim)
        else:
            self.attn = None

        # LSTM core
        if n_qubits > 0:
            self.lstm_cell = QuantumLSTMCell(embedding_dim, hidden_dim, n_qubits)
        else:
            # Fall back to the classical cell for compatibility
            self.lstm_cell = QuantumLSTMCell(embedding_dim, hidden_dim, 1)

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)

        # Fraud block
        embeds = self.fraud_block(embeds)

        # Self‑attention block
        if self.attn is not None:
            embeds = self.attn(embeds)

        # Initialise states
        batch_size = embeds.size(0)
        hx = torch.zeros(batch_size, self.hidden_dim, device=embeds.device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=embeds.device)

        outputs = []
        for x in embeds:
            hx, cx = self.lstm_cell(x, hx, cx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)

        tag_logits = self.hidden2tag(outputs)
        return F.log_softmax(tag_logits, dim=-1)


__all__ = ["HybridQLSTM", "FraudLayerParameters", "QuantumSelfAttention"]
