"""Hybrid quantum LSTM model with convolutional feature extraction.

This module defines a `HybridQLSTMQuantum` class that combines a
quantum convolution filter, a classical autoencoder, and a
quantum‑enhanced LSTM cell.  The design follows the structure of
the original `QLSTM.py` while integrating the quantum
convolution (QuanvCircuit) and the autoencoder from the other
reference pairs.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit import execute
import torchquantum as tq
import torchquantum.functional as tqf

# --------------------------------------------------------------------------- #
# Autoencoder utilities (from Autoencoder.py)
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
    """Standard MLP autoencoder."""
    def __init__(self, cfg: AutoencoderConfig):
        super().__init__()
        encoder_layers = []
        in_dim = cfg.input_dim
        for hidden in cfg.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                encoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = cfg.latent_dim
        for hidden in reversed(cfg.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                decoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(x))

def Autoencoder(input_dim: int,
                *, latent_dim: int = 32,
                hidden_dims: tuple[int, int] = (128, 64),
                dropout: float = 0.1) -> AutoencoderNet:
    """Convenience factory mirroring the original helper."""
    cfg = AutoencoderConfig(input_dim, latent_dim, hidden_dims, dropout)
    return AutoencoderNet(cfg)

# --------------------------------------------------------------------------- #
# Quantum convolution filter (from Conv.py)
# --------------------------------------------------------------------------- #
class QuanvCircuit:
    """Quantum filter used for quanvolution layers."""
    def __init__(self, kernel_size: int, backend=None, shots: int = 100, threshold: float = 0.5):
        self.n_qubits = kernel_size ** 2
        self._circuit = QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += qiskit.circuit.random.random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()
        self.backend = backend or AerSimulator()
        self.shots = shots
        self.threshold = threshold

    def run(self, data: np.ndarray) -> float:
        """Run the quantum circuit on classical data.

        Parameters
        ----------
        data : np.ndarray
            2D array with shape (kernel_size, kernel_size).

        Returns
        -------
        float
            Average probability of measuring |1> across qubits.
        """
        data = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0.0
            param_binds.append(bind)

        job = execute(self._circuit,
                      self.backend,
                      shots=self.shots,
                      parameter_binds=param_binds)
        result = job.result().get_counts(self._circuit)

        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val

        return counts / (self.shots * self.n_qubits)

# --------------------------------------------------------------------------- #
# Quantum LSTM implementation (from QLSTM.py)
# --------------------------------------------------------------------------- #
class QLSTM(nn.Module):
    """LSTM cell where gates are realised by small quantum circuits."""
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

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
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

    def forward(self, inputs: torch.Tensor, states: tuple[torch.Tensor, torch.Tensor] | None = None):
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

    def _init_states(self,
                     inputs: torch.Tensor,
                     states: tuple[torch.Tensor, torch.Tensor] | None) -> tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

class LSTMTagger(nn.Module):
    """Sequence tagging model that can switch between classical and quantum LSTM."""
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int,
                 tagset_size: int, n_qubits: int = 0) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

# --------------------------------------------------------------------------- #
# Hybrid quantum model integrating all components
# --------------------------------------------------------------------------- #
class HybridQLSTMQuantum(nn.Module):
    """Quantum‑enhanced hybrid LSTM that uses a quantum convolution
    filter, a classical autoencoder, and a quantum LSTM cell."""
    def __init__(self, embedding_dim: int, hidden_dim: int,
                 vocab_size: int, tagset_size: int,
                 latent_dim: int = 32, n_qubits: int = 4,
                 conv_kernel: int = 2, backend=None, shots: int = 100,
                 threshold: float = 0.5):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Quantum convolution filter
        self.conv_filter = QuanvCircuit(conv_kernel,
                                        backend=backend,
                                        shots=shots,
                                        threshold=threshold)

        # Classical autoencoder on scalar features
        self.autoencoder = Autoencoder(1,
                                       latent_dim=latent_dim,
                                       hidden_dims=(64, 32),
                                       dropout=0.1)

        # Quantum LSTM on the latent representation
        self.lstm = QLSTM(latent_dim, hidden_dim, n_qubits=n_qubits)

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        sentence : torch.LongTensor
            Tensor of token indices with shape (seq_len, batch).

        Returns
        -------
        torch.Tensor
            Log‑probabilities for each tag.
        """
        embeds = self.word_embeddings(sentence)          # (seq_len, batch, embed_dim)
        seq_len, batch, _ = embeds.shape

        # Convolutional feature extraction per token
        kernel = int(np.sqrt(self.embedding_dim))
        if kernel ** 2!= self.embedding_dim:
            raise ValueError("embedding_dim must be a perfect square for the quantum conv filter.")
        features = []
        for x in embeds.view(seq_len * batch, -1):
            arr = x.detach().cpu().numpy().reshape(kernel, kernel)
            feat = self.conv_filter.run(arr)
            features.append(feat)
        feat_tensor = torch.tensor(features, dtype=torch.float32,
                                   device=embeds.device).view(seq_len, batch, 1)

        # Autoencoding of scalar features
        latents = self.autoencoder.encode(feat_tensor.view(seq_len * batch, -1)).view(seq_len, batch, -1)

        lstm_out, _ = self.lstm(latents)
        tag_logits = self.hidden2tag(lstm_out.view(seq_len * batch, -1))
        return F.log_softmax(tag_logits, dim=1).view(seq_len, batch, -1)

__all__ = ["HybridQLSTMQuantum", "Autoencoder", "AutoencoderConfig",
           "AutoencoderNet", "QLSTM", "LSTMTagger"]
