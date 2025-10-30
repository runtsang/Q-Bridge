"""Quantum‑enhanced LSTM and classifier implemented with Qiskit.

This module exposes a :class:`QLSTM` that replaces the linear gate
computations of a classical LSTM with variational quantum circuits.
The :class:`LSTMTagger` wraps the quantum LSTM and feeds the hidden
state through a variational classifier circuit to produce tag logits.
"""

from __future__ import annotations

from typing import Iterable, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.providers.aer import AerSimulator


def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, list[SparsePauliOp]]:
    """Construct a variational ansatz followed by Z‑measurements.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the classifier circuit.
    depth : int
        Number of variational layers.

    Returns
    -------
    circuit : QuantumCircuit
        The full parameterised circuit.
    encoding : list[ParameterVector]
        Parameter vectors used for data encoding.
    weights : list[ParameterVector]
        Parameter vectors for the variational layers.
    observables : list[SparsePauliOp]
        Observables used to extract the classification logits.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)

    # Data‑encoding layer
    for idx, qubit in enumerate(range(num_qubits)):
        circuit.rx(encoding[idx], qubit)

    # Variational layers
    weight_idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[weight_idx], qubit)
            weight_idx += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    # Observables (Pauli‑Z on each qubit)
    observables = [
        SparsePauliOp("Z" + "I" * (num_qubits - i - 1) + "I" * i)
        for i in range(num_qubits)
    ]

    return circuit, list(encoding), list(weights), observables


class QLayer:
    """Quantum layer that implements a single LSTM gate.

    The layer maps an ``n_qubits``‑dimensional input vector to a set of
    rotation angles on a small variational circuit.  The expectation
    values of Pauli‑Z on each qubit are returned as the gate activation.
    """

    def __init__(self, n_qubits: int, depth: int = 1):
        self.n_qubits = n_qubits
        self.depth = depth
        self.backend = AerSimulator(method="statevector")

        # Parameter vectors
        self.x_params = ParameterVector("x", n_qubits)
        self.theta_params = ParameterVector("theta", n_qubits * depth)

        # Build the base circuit
        self.circuit = QuantumCircuit(n_qubits)
        for i in range(n_qubits):
            self.circuit.rx(self.x_params[i], i)
        for d in range(depth):
            for i in range(n_qubits):
                self.circuit.ry(self.theta_params[d * n_qubits + i], i)
            for i in range(n_qubits - 1):
                self.circuit.cz(i, i + 1)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch, n_qubits)`` – raw gate vector.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, n_qubits)`` – expectation values of Pauli‑Z.
        """
        batch = x.shape[0]
        results = []

        for i in range(batch):
            # Bind data‑encoding parameters to the input vector
            bindings = {p: x[i, j].item() for j, p in enumerate(self.x_params)}
            # Randomly initialise variational parameters
            rand_vals = torch.rand(self.n_qubits * self.depth)
            bindings.update({p: rand_vals[idx].item() for idx, p in enumerate(self.theta_params)})

            bound_circuit = self.circuit.bind_parameters(bindings)
            job = execute(bound_circuit, self.backend)
            state = job.result().get_statevector(bound_circuit)

            # Compute expectation of Pauli‑Z on each qubit
            exp_vals = []
            for qubit in range(self.n_qubits):
                exp = 0.0
                for idx, amp in enumerate(state):
                    bitstring = format(idx, f"0{self.n_qubits}b")
                    parity = (-1) ** int(bitstring[qubit])
                    exp += parity * abs(amp) ** 2
                exp_vals.append(exp)
            results.append(exp_vals)

        return torch.tensor(results, dtype=torch.float32, device=x.device)


class QLSTM(nn.Module):
    """Quantum LSTM cell where each gate is realised by a :class:`QLayer`."""

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int, depth: int = 1) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        assert hidden_dim == n_qubits, "hidden_dim must equal n_qubits for quantum LSTM"
        self.depth = depth

        # Linear projections to map concatenated input+hidden to gate vectors
        self.forget_linear = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.input_linear = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.update_linear = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.output_linear = nn.Linear(input_dim + hidden_dim, n_qubits)

        # Quantum layers for each gate
        self.forget_gate = QLayer(n_qubits, depth)
        self.input_gate = QLayer(n_qubits, depth)
        self.update_gate = QLayer(n_qubits, depth)
        self.output_gate = QLayer(n_qubits, depth)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []

        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)

            f = torch.sigmoid(self.forget_gate(self.forget_linear(combined)))
            i = torch.sigmoid(self.input_gate(self.input_linear(combined)))
            g = torch.tanh(self.update_gate(self.update_linear(combined)))
            o = torch.sigmoid(self.output_gate(self.output_linear(combined)))

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
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx


class LSTMTagger(nn.Module):
    """Sequence tagging model that combines the quantum LSTM with a quantum classifier head."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int = 2,
        n_qubits: int = 4,
        classifier_depth: int = 2,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=hidden_dim, depth=classifier_depth)

        # Quantum classifier circuit
        self.classifier_circuit, self.encoding, self.weights, self.observables = build_classifier_circuit(
            num_qubits=n_qubits,
            depth=classifier_depth,
        )
        self.backend = AerSimulator(method="statevector")

        # Linear mapping from Z‑expectations to tag logits
        self.logit_mapper = nn.Linear(n_qubits, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        hidden = lstm_out.view(len(sentence), -1)

        # Project hidden state to circuit dimension (hidden_dim == n_qubits)
        gate_activations = self.lstm.forget_gate(hidden)

        # Map to tag logits
        tag_logits = self.logit_mapper(gate_activations)
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["QLSTM", "LSTMTagger", "build_classifier_circuit"]
