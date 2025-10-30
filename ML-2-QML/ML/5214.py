"""Hybrid LSTM with optional quantum gates and auto‑encoder embedding.

This module implements a drop‑in replacement for the classical QLSTM
found in the seed.  It exposes a single class, :class:`QLSTMHybrid`,
that supports three operational modes:

* ``mode='classical'`` – the gates are standard linear layers.
* ``mode='quantum'`` – each gate is a tiny quantum circuit implemented
  with torchquantum.  The circuit is parameterised by a trainable
  variational ansatz and the expectation value of Pauli‑Z is used as
  the gate output.
* ``mode='autoenc'`` – the input sequence is first compressed by a
  small autoencoder (AutoencoderNet) before being fed into either
  classical or quantum gates.

The API mirrors the original QLSTM for seamless replacement.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------
#  Auto‑encoder helper
# ----------------------------------------------------------------------
@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 16
    hidden_dims: Tuple[int, int] = (32, 16)
    dropout: float = 0.1


class AutoencoderNet(nn.Module):
    def __init__(self, cfg: AutoencoderConfig):
        super().__init__()
        encoder = []
        in_dim = cfg.input_dim
        for h in cfg.hidden_dims:
            encoder.append(nn.Linear(in_dim, h))
            encoder.append(nn.ReLU())
            if cfg.dropout > 0:
                encoder.append(nn.Dropout(cfg.dropout))
            in_dim = h
        encoder.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*encoder)

        decoder = []
        in_dim = cfg.latent_dim
        for h in reversed(cfg.hidden_dims):
            decoder.append(nn.Linear(in_dim, h))
            decoder.append(nn.ReLU())
            if cfg.dropout > 0:
                decoder.append(nn.Dropout(cfg.dropout))
            in_dim = h
        decoder.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*decoder)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

# ----------------------------------------------------------------------
#  Quantum gate layer (torchquantum)
# ----------------------------------------------------------------------
class QuantumGateLayer(nn.Module):
    """Small parameterised circuit that returns a single expectation value.

    The circuit consists of a GeneralEncoder followed by a trainable
    rotation on each wire and a chain of CNOTs.  The expectation of
    Pauli‑Z on the last wire is returned.
    """
    def __init__(self, n_wires: int):
        super().__init__()
        import torchquantum as tq
        import torchquantum.functional as tqf

        self.n_wires = n_wires
        self.tq = tq
        self.tqf = tqf
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [i], "func": "rx", "wires": [i]}
                for i in range(n_wires)
            ]
        )
        self.params = nn.ModuleList(
            [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qdev = self.tq.QuantumDevice(
            n_wires=self.n_wires, bsz=x.shape[0], device=x.device
        )
        self.encoder(qdev, x)
        for w, gate in enumerate(self.params):
            gate(qdev, wires=w)
        for w in range(self.n_wires - 1):
            self.tqf.cnot(qdev, wires=[w, w + 1])
        # last wire holds the output
        return self.measure(qdev)[:, -1].unsqueeze(-1)

# ----------------------------------------------------------------------
#  Hybrid LSTM
# ----------------------------------------------------------------------
class QLSTMHybrid(nn.Module):
    """Hybrid LSTM that can operate in classical, quantum, or auto‑encoded mode.

    Parameters
    ----------
    input_dim : int
        Size of the input vector at each timestep.
    hidden_dim : int
        Size of the hidden state.
    n_qubits : int
        Number of qubits used in the quantum gates.  Must be >0 if
        ``mode='quantum'``.
    mode : {'classical', 'quantum', 'autoenc'}
        Operational mode.
    autoencoder_cfg : Optional[AutoencoderConfig]
        Configuration for the autoencoder when ``mode='autoenc'``.  If
        ``None`` a default config is used.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int = 0,
        mode: str = "classical",
        autoencoder_cfg: Optional[AutoencoderConfig] = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.mode = mode

        if mode not in {"classical", "quantum", "autoenc"}:
            raise ValueError(f"Unsupported mode {mode!r}")

        # gating layers
        if mode == "classical":
            self.forget = nn.Linear(input_dim + hidden_dim, hidden_dim)
            self.input = nn.Linear(input_dim + hidden_dim, hidden_dim)
            self.update = nn.Linear(input_dim + hidden_dim, hidden_dim)
            self.output = nn.Linear(input_dim + hidden_dim, hidden_dim)
        else:
            # quantum gates
            self.forget = QuantumGateLayer(n_qubits)
            self.input = QuantumGateLayer(n_qubits)
            self.update = QuantumGateLayer(n_qubits)
            self.output = QuantumGateLayer(n_qubits)

            # linear projections to qubits
            self.forget_proj = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.input_proj = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.update_proj = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.output_proj = nn.Linear(input_dim + hidden_dim, n_qubits)

        # optional auto‑encoder
        if mode == "autoenc":
            cfg = autoencoder_cfg or AutoencoderConfig(input_dim=input_dim)
            self.autoencoder = AutoencoderNet(cfg)
            # after encoding the dimension reduces to latent_dim
            enc_dim = cfg.latent_dim
            # recompute projections for the encoded dimension
            if mode == "quantum":
                self.forget_proj = nn.Linear(enc_dim + hidden_dim, n_qubits)
                self.input_proj = nn.Linear(enc_dim + hidden_dim, n_qubits)
                self.update_proj = nn.Linear(enc_dim + hidden_dim, n_qubits)
                self.output_proj = nn.Linear(enc_dim + hidden_dim, n_qubits)
            else:
                self.forget = nn.Linear(enc_dim + hidden_dim, hidden_dim)
                self.input = nn.Linear(enc_dim + hidden_dim, hidden_dim)
                self.update = nn.Linear(enc_dim + hidden_dim, hidden_dim)
                self.output = nn.Linear(enc_dim + hidden_dim, hidden_dim)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch, self.hidden_dim, device=device)
        cx = torch.zeros(batch, self.hidden_dim, device=device)
        return hx, cx

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Run the hybrid LSTM over a sequence.

        Parameters
        ----------
        inputs : Tensor of shape (seq_len, batch, feature)
            Input sequence.
        states : optional tuple (hx, cx)
            Initial hidden and cell states.

        Returns
        -------
        outputs : Tensor of shape (seq_len, batch, hidden_dim)
            Hidden states at each timestep.
        final_state : tuple (hx, cx)
            Final hidden and cell states.
        """
        if self.mode == "autoenc":
            # compress each timestep
            seq_len, batch, _ = inputs.shape
            flat = inputs.reshape(seq_len * batch, -1)
            compressed = self.autoencoder.encode(flat)
            compressed = compressed.view(seq_len, batch, -1)
            inputs = compressed

        hx, cx = self._init_states(inputs, states)
        outputs = []

        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            if self.mode == "classical":
                f = torch.sigmoid(self.forget(combined))
                i = torch.sigmoid(self.input(combined))
                g = torch.tanh(self.update(combined))
                o = torch.sigmoid(self.output(combined))
            else:
                # quantum mode
                f = torch.sigmoid(self.forget(self.forget_proj(combined)))
                i = torch.sigmoid(self.input(self.input_proj(combined)))
                g = torch.tanh(self.update(self.update_proj(combined)))
                o = torch.sigmoid(self.output(self.output_proj(combined)))

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))

        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

__all__ = ["QLSTMHybrid", "AutoencoderNet", "AutoencoderConfig"]
