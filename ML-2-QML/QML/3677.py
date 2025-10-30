"""Quantum‑enhanced classifier and LSTM tagger using Qiskit.

The module defines:
  * :class:`HybridClassifier` – a quantum circuit that mirrors the
    classical feed‑forward residual network builder interface.
  * :class:`QuantumQLSTM` – a quantum LSTM cell built from small
    parameterised Qiskit circuits.
  * :class:`LSTMTagger` – sequence tagging that can switch between the
    quantum LSTM and a classical LSTM.

The build_classifier_circuit static method returns a tuple
(circuit, encoding, weights, observables) exactly as the
original anchor file.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
import numpy as np

class HybridClassifier:
    """Quantum circuit that mirrors the classical feed‑forward network."""

    @staticmethod
    def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
        """Return a parameterised quantum circuit, encoding and weight lists, and observables."""

        # Encoding of raw features
        encoding = ParameterVector("x", num_qubits)
        # Variational parameters per layer
        weights = ParameterVector("theta", num_qubits * depth)

        circuit = QuantumCircuit(num_qubits)
        # Encode input
        for i, param in enumerate(encoding):
            circuit.rx(param, i)

        # Build layered ansatz
        idx = 0
        for _ in range(depth):
            # Apply layer of parameterised rotations
            for qubit in range(num_qubits):
                circuit.ry(weights[idx], qubit)
                idx += 1
            # Entangle neighbouring qubits with CZ
            for qubit in range(num_qubits - 1):
                circuit.cz(qubit, qubit + 1)

        # Measurement observables – Z on each qubit
        observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]

        return circuit, list(encoding), list(weights), observables

class QuantumQLSTM:
    """Quantum LSTM cell built from small parameterised Qiskit circuits."""

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Define parameterised circuits for each gate
        self.forget_circ = self._gate_circuit("forget")
        self.input_circ = self._gate_circuit("input")
        self.update_circ = self._gate_circuit("update")
        self.output_circ = self._gate_circuit("output")

        # Classical linear layers to map linear combos to qubit states
        self.linear_forget = np.random.randn(input_dim + hidden_dim, n_qubits)
        self.linear_input = np.random.randn(input_dim + hidden_dim, n_qubits)
        self.linear_update = np.random.randn(input_dim + hidden_dim, n_qubits)
        self.linear_output = np.random.randn(input_dim + hidden_dim, n_qubits)

    def _gate_circuit(self, name: str) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        # Example: each qubit gets an RX rotation followed by a CNOT network
        for q in range(self.n_qubits):
            qc.rx(0.0, q)  # placeholder parameter
        for q in range(self.n_qubits - 1):
            qc.cx(q, q + 1)
        return qc

    def forward(self, inputs: np.ndarray, states: Tuple[np.ndarray, np.ndarray] | None = None) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs:
            combined = np.concatenate([x, hx], axis=0)
            # Map to qubit parameters
            f_params = self.linear_forget @ combined
            i_params = self.linear_input @ combined
            g_params = self.linear_update @ combined
            o_params = self.linear_output @ combined

            # Run circuits (placeholder)
            f = np.tanh(f_params)  # placeholder for measurement
            i = np.tanh(i_params)
            g = np.tanh(g_params)
            o = np.tanh(o_params)

            cx = f * cx + i * g
            hx = o * np.tanh(cx)
            outputs.append(hx.copy())
        return np.stack(outputs), (hx, cx)

    def _init_states(self, inputs: np.ndarray, states: Tuple[np.ndarray, np.ndarray] | None) -> Tuple[np.ndarray, np.ndarray]:
        if states is not None:
            return states
        batch_size = inputs.shape[0]
        hx = np.zeros((batch_size, self.hidden_dim))
        cx = np.zeros((batch_size, self.hidden_dim))
        return hx, cx

class LSTMTagger:
    """Sequence tagging that can use either the quantum LSTM or a classical LSTM."""

    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int, tagset_size: int, n_qubits: int = 0):
        self.hidden_dim = hidden_dim
        self.word_embeddings = np.random.randn(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QuantumQLSTM(embedding_dim, hidden_dim, n_qubits)
        else:
            self.lstm = None  # placeholder for classical LSTM

        self.hidden2tag = np.random.randn(hidden_dim, tagset_size)

    def forward(self, sentence: np.ndarray) -> np.ndarray:
        embeds = self.word_embeddings[sentence]
        if self.lstm:
            lstm_out, _ = self.lstm.forward(embeds)
        else:
            # Placeholder classical LSTM
            lstm_out = np.tanh(embeds)
        tag_logits = lstm_out @ self.hidden2tag
        return tag_logits

__all__ = ["HybridClassifier", "QuantumQLSTM", "LSTMTagger"]
