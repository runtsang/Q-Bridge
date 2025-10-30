"""Quantum LSTM with optional quantum auto‑encoder preprocessing.

The module mirrors the classical variant but uses quantum circuits for
the LSTM gates.  An optional quantum auto‑encoder (SamplerQNN) can be
applied to the input data before it reaches the LSTM, providing a
quantum feature extraction stage that is analogous to the classical
auto‑encoder in the ML module.

Classes
-------
QLSTM : nn.Module
    Quantum LSTM cell that can optionally prepend a quantum auto‑encoder.
LSTMTagger : nn.Module
    Sequence tagging model that uses QLSTM and an optional quantum auto‑encoder.
QuantumAutoencoder : nn.Module
    Simple variational auto‑encoder built with Qiskit and a SamplerQNN.
"""

from __future__ import annotations

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Quantum LSTM gates – torchquantum
import torchquantum as tq
import torchquantum.functional as tqf

# Quantum auto‑encoder – qiskit
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN


class QuantumAutoencoder(nn.Module):
    """A lightweight variational auto‑encoder using a SamplerQNN.

    The circuit consists of a RealAmplitudes ansatz followed by a
    swap‑test that projects the latent subspace onto a single qubit.
    The output of the sampler is interpreted as a probability
    distribution over the two basis states, which can be used as a
    learned feature vector.

    Parameters
    ----------
    num_latent : int, default 3
        Number of latent qubits.
    num_trash : int, default 2
        Number of auxiliary qubits used in the swap‑test.
    reps : int, default 5
        Number of repetitions of the ansatz.
    """

    def __init__(self, num_latent: int = 3, num_trash: int = 2, reps: int = 5) -> None:
        super().__init__()
        self.num_latent = num_latent
        self.num_trash = num_trash
        self.reps = reps

        self.sampler = Sampler()
        self.circuit = self._build_circuit()

        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters,
            interpret=lambda x: x,
            output_shape=2,
            sampler=self.sampler,
        )

    def _build_circuit(self) -> QuantumCircuit:
        qr = QuantumRegister(self.num_latent + 2 * self.num_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        circuit = QuantumCircuit(qr, cr)

        # Ansatz for latent + trash qubits
        circuit.compose(
            RealAmplitudes(self.num_latent + self.num_trash, reps=self.reps),
            range(0, self.num_latent + self.num_trash),
            inplace=True,
        )
        circuit.barrier()

        aux = self.num_latent + 2 * self.num_trash
        circuit.h(aux)
        for i in range(self.num_trash):
            circuit.cswap(aux, self.num_latent + i, self.num_latent + self.num_trash + i)

        circuit.h(aux)
        circuit.measure(aux, cr[0])

        return circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the auto‑encoder.

        The input tensor is interpreted as a batch of parameter vectors
        for the ansatz.  In practice the parameters would be trainable;
        here they are left as a placeholder.
        """
        return self.qnn(x)


class QLSTM(nn.Module):
    """Quantum LSTM cell with optional quantum auto‑encoder preprocessing.

    Parameters
    ----------
    input_dim : int
        Size of the input vectors (pre‑processed by the auto‑encoder
        if supplied).
    hidden_dim : int
        Size of the hidden state.
    n_qubits : int
        Number of qubits used in each gate circuit.
    autoencoder : Optional[QuantumAutoencoder]
        If provided, the quantum auto‑encoder is applied to all inputs
        before the LSTM gates are computed.
    """

    class QLayer(tq.QuantumModule):
        """Small variational circuit used as a gate."""

        def __init__(self, n_wires: int):
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
            qdev = tq.QuantumDevice(
                n_wires=self.n_wires, bsz=x.shape[0], device=x.device
            )
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            for wire in range(self.n_wires):
                if wire == self.n_wires - 1:
                    tqf.cnot(qdev, wires=[wire, 0])
                else:
                    tqf.cnot(qdev, wires=[wire, wire + 1])
            return self.measure(qdev)

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int,
        autoencoder: Optional[QuantumAutoencoder] = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.autoencoder = autoencoder

        effective_input_dim = (
            self.autoencoder.circuit.num_qubits if self.autoencoder is not None else input_dim
        )

        self.forget = self.QLayer(n_qubits)
        self.input = self.QLayer(n_qubits)
        self.update = self.QLayer(n_qubits)
        self.output = self.QLayer(n_qubits)

        self.linear_forget = nn.Linear(effective_input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(effective_input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(effective_input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(effective_input_dim + hidden_dim, n_qubits)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.autoencoder is not None:
            # Use the quantum auto‑encoder to transform the inputs.
            inputs = self.autoencoder(inputs)

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
    """Sequence tagging model that uses :class:`QLSTM` with optional quantum auto‑encoder.

    Parameters
    ----------
    embedding_dim : int
        Size of word embeddings.
    hidden_dim : int
        Size of the LSTM hidden state.
    vocab_size : int
        Vocabulary size.
    tagset_size : int
        Number of target tags.
    n_qubits : int, optional
        Number of qubits for the quantum LSTM gates.
    autoencoder : Optional[QuantumAutoencoder]
        Quantum auto‑encoder used for feature extraction.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        autoencoder: Optional[QuantumAutoencoder] = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        if n_qubits > 0:
            self.lstm = QLSTM(
                embedding_dim,
                hidden_dim,
                n_qubits=n_qubits,
                autoencoder=autoencoder,
            )
        else:
            raise NotImplementedError(
                "The classical LSTM branch is implemented in the ML module."
            )

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """Forward pass for sequence tagging.

        Parameters
        ----------
        sentence : torch.Tensor
            Tensor of word indices of shape ``(seq_len, batch)``.

        Returns
        -------
        torch.Tensor
            Log‑softmax over tag predictions of shape
            ``(seq_len, batch, tagset_size)``.
        """
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["QLSTM", "LSTMTagger", "QuantumAutoencoder"]
