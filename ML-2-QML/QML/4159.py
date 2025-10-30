"""Hybrid quantum LSTM with auto‑encoder preprocessing and quantum sampler‑based masking.

The quantum version replaces each LSTM gate with a small variational
circuit.  The input embeddings are first compressed with the same
auto‑encoder used in the classical module.  A Qiskit ``SamplerQNN``
provides a 4‑dimensional mask that stochastically modulates the
forget, input, update and output gates.  The overall API matches the
original ``QLSTM`` so the tagger can switch between a classical
and a quantum cell with a single flag.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN as QiskitSamplerQNN

# --------------------------------------------------------------------------- #
# Auto‑encoder utilities (identical to the classical module)
# --------------------------------------------------------------------------- #
class AutoencoderConfig:
    """Configuration for :class:`AutoencoderNet`."""
    def __init__(self, input_dim: int, latent_dim: int = 32,
                 hidden_dims: tuple[int, int] = (128, 64), dropout: float = 0.1):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout


class AutoencoderNet(nn.Module):
    """A lightweight fully‑connected auto‑encoder."""
    def __init__(self, cfg: AutoencoderConfig):
        super().__init__()
        encoder_layers = []
        in_dim = cfg.input_dim
        for hidden in cfg.hidden_dims:
            encoder_layers.extend([nn.Linear(in_dim, hidden),
                                   nn.ReLU(),
                                   nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()])
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = cfg.latent_dim
        for hidden in reversed(cfg.hidden_dims):
            decoder_layers.extend([nn.Linear(in_dim, hidden),
                                   nn.ReLU(),
                                   nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()])
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


# --------------------------------------------------------------------------- #
# Quantum LSTM cell
# --------------------------------------------------------------------------- #
class QLSTM(nn.Module):
    """Quantum LSTM cell where each gate is a small variational circuit,
    with auto‑encoder feature extraction and a quantum sampler for
    stochastic masking."""
    class QLayer(tq.QuantumModule):
        """A generic 4‑wire variational layer used for each LSTM gate."""
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            # Simple feature encoder
            self.encoder = tq.GeneralEncoder([
                {"input_idx": [0], "func": "rx", "wires": [0]},
                {"input_idx": [1], "func": "rx", "wires": [1]},
                {"input_idx": [2], "func": "rx", "wires": [2]},
                {"input_idx": [3], "func": "rx", "wires": [3]},
            ])
            self.params = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            qdev = tq.QuantumDevice(n_wires=self.n_wires,
                                    bsz=x.shape[0],
                                    device=x.device)
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            for wire in range(self.n_wires):
                if wire == self.n_wires - 1:
                    tqf.cnot(qdev, wires=[wire, 0])
                else:
                    tqf.cnot(qdev, wires=[wire, wire + 1])
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int,
                 n_qubits: int, latent_dim: int | None = None):
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

        # Auto‑encoder for compressing classical features
        latent = latent_dim or hidden_dim
        self.autoencoder = AutoencoderNet(AutoencoderConfig(input_dim,
                                                            latent_dim=latent))
        # Quantum sampler for stochastic masking
        self.sampler = Sampler()
        input_params = ParameterVector("input", n_qubits)
        weight_params = ParameterVector("weight", n_qubits)
        qc = QuantumCircuit(n_qubits)
        for i in range(n_qubits):
            qc.rx(input_params[i], i)
            qc.rx(weight_params[i], i)
        self.qsampler = QiskitSamplerQNN(circuit=qc,
                                         input_params=input_params,
                                         weight_params=weight_params,
                                         sampler=self.sampler,
                                         output_shape=4)

    def _init_states(self, inputs: torch.Tensor,
                     states: tuple[torch.Tensor, torch.Tensor] | None = None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))

    def forward(self, inputs: torch.Tensor,
                states: tuple[torch.Tensor, torch.Tensor] | None = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            # Compress input via auto‑encoder
            comp = self.autoencoder.encode(x)
            combined = torch.cat([comp, hx], dim=1)

            f = torch.sigmoid(self.forget(self.linear_forget(combined)))
            i = torch.sigmoid(self.input(self.linear_input(combined)))
            g = torch.tanh(self.update(self.linear_update(combined)))
            o = torch.sigmoid(self.output(self.linear_output(combined)))

            # Stochastic mask from quantum sampler
            mask_np = self.qsampler.sample()
            mask = torch.tensor(mask_np, dtype=torch.float32,
                                device=inputs.device)
            f *= mask[0]
            i *= mask[1]
            g *= mask[2]
            o *= mask[3]

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        return torch.cat(outputs, dim=0), (hx, cx)


# --------------------------------------------------------------------------- #
# Tagger that can switch between the hybrid quantum LSTM and a vanilla nn.LSTM
# --------------------------------------------------------------------------- #
class LSTMTagger(nn.Module):
    """Sequence tagging model that can switch between the hybrid quantum LSTM
    and a vanilla ``nn.LSTM``."""
    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 n_qubits: int = 0,
                 latent_dim: int | None = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim,
                              n_qubits=n_qubits, latent_dim=latent_dim)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["QLSTM", "LSTMTagger", "AutoencoderNet", "AutoencoderConfig"]
