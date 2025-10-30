"""Quantum‑enhanced LSTM using photonic circuits.

The module implements a `QLSTM` that mirrors the classical interface but
replaces each gate with a small photonic circuit built from
`FraudLayerParameters`.  The circuits are executed on a StrawberryFields
backend and the measurement results are fed back into the LSTM recurrence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import strawberryfields as sf
from strawberryfields import ops


@dataclass
class FraudLayerParameters:
    """Parameters that define a photonic layer used in the quantum gates."""

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


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> sf.Program:
    """Create a StrawberryFields program that implements a fraud‑detection style layer."""
    program = sf.Program(2)
    with program.context as q:
        _apply_layer(q, input_params, clip=False)
        for layer in layers:
            _apply_layer(q, layer, clip=True)
    return program


def _apply_layer(
    modes: Iterable, params: FraudLayerParameters, *, clip: bool
) -> None:
    ops.BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
    for i, phase in enumerate(params.phases):
        ops.Rgate(phase) | modes[i]
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        ops.Sgate(r if not clip else _clip(r, 5), phi) | modes[i]
    ops.BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
    for i, phase in enumerate(params.phases):
        ops.Rgate(phase) | modes[i]
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        ops.Dgate(r if not clip else _clip(r, 5), phi) | modes[i]
    for i, k in enumerate(params.kerr):
        ops.Kgate(k if not clip else _clip(k, 1)) | modes[i]


class QLayer(nn.Module):
    """Quantum layer that encodes an input vector into a photonic circuit."""

    def __init__(
        self,
        input_dim: int,
        n_wires: int,
        params: FraudLayerParameters,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.n_wires = n_wires
        self.linear_in = nn.Linear(input_dim, n_wires)
        self.program = build_fraud_detection_program(params, [])
        self.backend = sf.backends.SecAccelerator()
        self.backend.init(n_wires=n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Run the photonic circuit for each sample in the batch."""
        x_w = self.linear_in(x)  # (batch, n_wires)
        outputs = []
        for i in range(x_w.shape[0]):
            prog = self.program.copy()
            # Encode the input as displacement gates
            for w in range(self.n_wires):
                ops.Dgate(x_w[i, w].item()) | w
            result = self.backend.run(prog, shots=1)
            # Convert the sampled state into a tensor
            sample = result.state.sample()
            outputs.append(sample)
        return torch.tensor(outputs, dtype=torch.float32, device=x.device)


class QLSTM(nn.Module):
    """Quantum LSTM cell that uses photonic gates for the four LSTM gates."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int,
        gate_params: Optional[Iterable[FraudLayerParameters]] = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Default parameters for the gates if none are supplied
        default_params = [
            FraudLayerParameters(
                bs_theta=0.0,
                bs_phi=0.0,
                phases=(0.0, 0.0),
                squeeze_r=(0.0, 0.0),
                squeeze_phi=(0.0, 0.0),
                displacement_r=(0.0, 0.0),
                displacement_phi=(0.0, 0.0),
                kerr=(0.0, 0.0),
            )
            for _ in range(4)
        ]

        params_iter = gate_params or default_params
        self.forget = QLayer(input_dim + hidden_dim, n_qubits, next(params_iter))
        self.input = QLayer(input_dim + hidden_dim, n_qubits, next(params_iter))
        self.update = QLayer(input_dim + hidden_dim, n_qubits, next(params_iter))
        self.output = QLayer(input_dim + hidden_dim, n_qubits, next(params_iter))

        # Classical linear maps from the concatenated input to the circuit
        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

        # Map the quantum output back to the hidden dimension
        self.to_hidden = nn.Linear(n_qubits, hidden_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(
                self.forget(
                    self.linear_forget(combined)
                )
            )
            i = torch.sigmoid(
                self.input(
                    self.linear_input(combined)
                )
            )
            g = torch.tanh(
                self.update(
                    self.linear_update(combined)
                )
            )
            o = torch.sigmoid(
                self.output(
                    self.linear_output(combined)
                )
            )
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )


class LSTMTagger(nn.Module):
    """Sequence tagging model that can switch between the quantum and classical LSTM."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        gate_params: Optional[Iterable[FraudLayerParameters]] = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(
                embedding_dim,
                hidden_dim,
                n_qubits=n_qubits,
                gate_params=gate_params,
            )
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(
            embeds.view(len(sentence), 1, -1)
        )
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["QLSTM", "LSTMTagger"]
