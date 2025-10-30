"""
Quantum‑enhanced LSTM tagger with a variational quantum autoencoder.
"""

from __future__ import annotations

from typing import Iterable, Tuple, Optional

import torch
from torch import nn
import torch.nn.functional as F
import qiskit as qk
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals
from qiskit.primitives import StatevectorSampler


class QuantumAutoencoder(nn.Module):
    """
    Variational quantum autoencoder using a RealAmplitudes ansatz
    and a swap‑test based decoding circuit.
    """
    def __init__(self, input_dim: int, latent_dim: int, trash_dim: int = 2) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.trash_dim = trash_dim
        self.circuit = self._build_circuit()

        self.sampler = StatevectorSampler()
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters,
            interpret=lambda x: x,
            output_shape=2,
            sampler=self.sampler,
        )

    def _build_circuit(self) -> QuantumCircuit:
        qr = QuantumRegister(self.latent_dim + 2 * self.trash_dim + 1, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)

        # Encode stage
        qc.compose(RealAmplitudes(self.latent_dim + self.trash_dim, reps=5),
                   range(0, self.latent_dim + self.trash_dim), inplace=True)
        qc.barrier()

        # Auxiliary qubit for swap test
        aux = self.latent_dim + 2 * self.trash_dim
        qc.h(aux)
        for i in range(self.trash_dim):
            qc.cswap(aux, self.latent_dim + i, self.latent_dim + self.trash_dim + i)
        qc.h(aux)
        qc.measure(aux, cr[0])
        return qc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input is ignored; weights are trainable parameters.
        return self.qnn.forward(x)


class QLSTM(nn.Module):
    """
    Quantum LSTM cell where gates are realised by small quantum circuits.
    """
    class QLayer(nn.Module):
        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.encoder = qk.circuit.library.RealAmplitudes(n_wires, reps=2)
            self.params = nn.ParameterList(
                [nn.Parameter(torch.rand(1)) for _ in range(n_wires)]
            )
            self.measure = qk.circuit.library.MeasureAll(qk.circuit.library.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            dev = qk.QuantumCircuit(self.n_wires)
            dev.compose(self.encoder, inplace=True)
            for i, p in enumerate(self.params):
                dev.rx(p, i)
            dev = dev.compose(self.measure, inplace=True)
            # Simulate on CPU backend
            result = qk.execute(dev, backend="qasm_simulator", shots=1).get_counts()
            return torch.tensor(list(result.values()), dtype=torch.float32)

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

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
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


class LSTMTagger(nn.Module):
    """
    Sequence tagging model that can switch between classical and
    quantum LSTM cells.  A quantum autoencoder compresses the
    embeddings before the recurrent layer.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        autoencoder: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.autoencoder = autoencoder
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        if self.autoencoder is not None:
            embeds = self.autoencoder(embeds)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["QLSTM", "LSTMTagger", "QuantumAutoencoder"]
