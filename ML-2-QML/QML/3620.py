"""Quantum‑enhanced LSTM with a Qiskit variational classifier head.

The LSTM gates are implemented with torchquantum modules,
while the final classification layer is a Qiskit variational
circuit that can be simulated on a classical backend.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.providers.aer import AerSimulator


def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, list[SparsePauliOp]]:
    """Construct a variational ansatz for the classifier.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit (equal to the hidden dimension).
    depth : int
        Depth of the variational layers.

    Returns
    -------
    QuantumCircuit
        The variational circuit with parametrized rotations and entangling gates.
    Iterable
        Encoding parameters (rx angles).
    Iterable
        Variational parameters (ry angles).
    list[SparsePauliOp]
        Observable Pauli strings used for measurement.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    qc = QuantumCircuit(num_qubits)
    for idx, qubit in enumerate(range(num_qubits)):
        qc.rx(encoding[idx], qubit)

    w_idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            qc.ry(weights[w_idx], qubit)
            w_idx += 1
        for qubit in range(num_qubits - 1):
            qc.cz(qubit, qubit + 1)

    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return qc, list(encoding), list(weights), observables


class QLayer(tq.QuantumModule):
    """Small quantum module that implements a single gate of the LSTM.

    The module encodes a vector of gate parameters into a
    quantum state, applies a layer of RX rotations and a
    CNOT chain, and measures all qubits with Pauli‑Z.
    """

    def __init__(self, n_wires: int) -> None:
        super().__init__()
        self.n_wires = n_wires
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
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
        self.encoder(qdev, x)
        for wire, gate in enumerate(self.params):
            gate(qdev, wires=wire)
        for wire in range(self.n_wires - 1):
            tqf.cnot(qdev, wires=[wire, wire + 1])
        tqf.cnot(qdev, wires=[self.n_wires - 1, 0])
        return self.measure(qdev)


class QLSTM(nn.Module):
    """Quantum‑enhanced LSTM cell with a Qiskit variational classifier.

    The gates are realized by :class:`QLayer`.  After the hidden
    sequence is produced, each hidden vector is encoded into a
    quantum circuit and the expectation values of Z on every qubit
    are used as logits.  The circuit is simulated with Aer.
    """

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int, depth: int = 2) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Quantum gates for the LSTM
        self.forget = QLayer(n_qubits)
        self.input = QLayer(n_qubits)
        self.update = QLayer(n_qubits)
        self.output = QLayer(n_qubits)

        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

        # Variational classifier
        self.classifier_circuit, self.encoding, self.weight_sizes, self.observables = build_classifier_circuit(
            num_qubits=hidden_dim, depth=depth
        )
        self.quantum_sim = AerSimulator(method="statevector")

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        hx, cx = self._init_states(inputs, states)
        outputs: list[torch.Tensor] = []

        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(self.linear_forget(combined)))
            i = torch.sigmoid(self.input(self.linear_input(combined)))
            g = torch.tanh(self.update(self.linear_update(combined)))
            o = torch.sigmoid(self.output(self.linear_output(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))

        hidden_seq = torch.cat(outputs, dim=0)                      # (seq_len, batch, hidden_dim)
        logits = self.quantum_classifier(hidden_seq)
        return hidden_seq, (hx, cx), logits

    def quantum_classifier(self, hidden: torch.Tensor) -> torch.Tensor:
        """Run the variational classifier on each hidden vector.

        The hidden tensor is expected to be of shape
        (seq_len, batch, hidden_dim).  For simplicity we
        batch‑process all vectors using a loop over the
        sequence dimension and a nested loop over the batch.
        """
        seq_len, batch, hidden_dim = hidden.shape
        logits_list: list[torch.Tensor] = []

        for t in range(seq_len):
            batch_logits: list[torch.Tensor] = []
            for b in range(batch):
                params = hidden[t, b]                                 # (hidden_dim,)
                bound = {self.encoding[i]: params[i].item() for i in range(hidden_dim)}
                qc_b = self.classifier_circuit.bind_parameters(bound)
                result = self.quantum_sim.run(qc_b, shots=0).result()
                state = result.get_statevector()
                exps = [state.expectation_value(obs).real for obs in self.observables]
                logits = torch.tensor(exps, dtype=torch.float32)
                batch_logits.append(logits)
            logits_list.append(torch.stack(batch_logits))

        return torch.stack(logits_list)  # (seq_len, batch, hidden_dim)

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
    """Sequence tagging model that can switch between classical and quantum LSTM.

    Parameters
    ----------
    embedding_dim : int
        Dimension of token embeddings.
    hidden_dim : int
        Hidden dimension of the LSTM.
    vocab_size : int
        Number of distinct tokens.
    tagset_size : int
        Number of tags in the output space.
    n_qubits : int, default 0
        If >0, a :class:`QLSTM` (quantum) is instantiated; otherwise a
        standard :class:`nn.LSTM` is used.
    depth : int, default 2
        Depth of the variational classifier.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        depth: int = 2,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits, depth=depth)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, states, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["QLSTM", "LSTMTagger"]
