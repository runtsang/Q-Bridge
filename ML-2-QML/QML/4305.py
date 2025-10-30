"""Quantum‑enhanced autoencoder using Qiskit SamplerQNN and TorchQuantum QLSTM."""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RealAmplitudes
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit_machine_learning.utils import algorithm_globals

import torchquantum as tq
import torchquantum.functional as tqf

algorithm_globals.random_seed = 42

# ------------------------------------------------------------
# Quantum LSTM (from reference pair 3)
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# Hybrid Autoencoder (quantum variant)
# ------------------------------------------------------------
class HybridAutoencoder(nn.Module):
    """Quantum autoencoder that encodes inputs via a SamplerQNN, processes sequences with a TorchQuantum QLSTM, and decodes back with a second SamplerQNN."""
    def __init__(self, input_dim: int, latent_dim: int = 32, hidden_dims: Tuple[int, int] = (128, 64), dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout

        # Encoder QNN
        self.encoder = self._make_encoder()

        # Quantum LSTM
        self.lstm = QLSTM(input_dim=latent_dim, hidden_dim=latent_dim, n_qubits=latent_dim)

        # Decoder QNN
        self.decoder = self._make_decoder()

    def _make_encoder(self):
        qc = QuantumCircuit(self.latent_dim)
        x = ParameterVector('x', self.input_dim)
        # Encode each input dimension into a qubit via RX
        for i in range(self.latent_dim):
            qc.rx(x[i % self.input_dim], i)
        ansatz = RealAmplitudes(self.latent_dim, reps=1)
        qc.append(ansatz, range(self.latent_dim))
        qc.barrier()
        def identity_interpret(x):
            return x
        return SamplerQNN(
            circuit=qc,
            input_params=list(x),
            weight_params=list(ansatz.parameters),
            interpret=identity_interpret,
            output_shape=self.latent_dim,
            sampler=Sampler(),
        )

    def _make_decoder(self):
        qc = QuantumCircuit(self.input_dim)
        z = ParameterVector('z', self.latent_dim)
        for i in range(self.input_dim):
            qc.rx(z[i % self.latent_dim], i)
        ansatz = RealAmplitudes(self.input_dim, reps=1)
        qc.append(ansatz, range(self.input_dim))
        qc.barrier()
        def identity_interpret(x):
            return x
        return SamplerQNN(
            circuit=qc,
            input_params=list(z),
            weight_params=list(ansatz.parameters),
            interpret=identity_interpret,
            output_shape=self.input_dim,
            sampler=Sampler(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor of shape (batch, seq_len, input_dim)
        Returns reconstructed tensor of the same shape.
        """
        batch, seq_len, _ = x.shape
        flat = x.reshape(batch * seq_len, self.input_dim)
        z = self.encoder(flat).to(x.device)
        z_seq = z.reshape(batch, seq_len, self.latent_dim)
        # QLSTM expects input shape (seq_len, batch, features)
        lstm_input = z_seq.permute(1, 0, 2)
        lstm_out, _ = self.lstm(lstm_input)
        lstm_out = lstm_out.permute(1, 0, 2).reshape(batch * seq_len, self.latent_dim)
        recon = self.decoder(lstm_out).to(x.device)
        recon = recon.reshape(batch, seq_len, self.input_dim)
        return recon

def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
) -> HybridAutoencoder:
    """Factory that returns a quantum :class:`HybridAutoencoder` instance."""
    return HybridAutoencoder(input_dim, latent_dim=latent_dim, hidden_dims=hidden_dims, dropout=dropout)

__all__ = ["Autoencoder", "HybridAutoencoder", "QLSTM"]
