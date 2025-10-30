"""Quantum‑enhanced LSTM with variational circuits for each gate.

This module implements :class:`HybridQLSTM`, a drop‑in replacement for
the classical LSTM that replaces the linear gate transformations with
small variational quantum circuits.  The design is inspired by the
original QML QLSTM and the QCNet hybrid head, but it is fully
self‑contained and exposes the same public API.

Key features
------------
- **Quantum gates** – each of the forget, input, update and output
  gates is realised by a variational circuit that outputs a vector of
  expectation values of Pauli Z on `n_qubits` qubits.
- **Batch evaluation** – a :class:`FastBaseEstimator` wrapper is
  provided to evaluate the circuit expectation values for a list of
  parameter sets.
- **Compatibility** – the class names `QLSTM` and `LSTMTagger` are
  kept as aliases so existing code continues to run unchanged.

The implementation uses Qiskit Aer for simulation and is written to
be easily extended to a real quantum back‑end.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from qiskit import QuantumCircuit, transpile, assemble
from qiskit.quantum_info import Statevector
from qiskit.providers import Backend
from qiskit.providers.aer import AerSimulator

# ----------------------------------------------------------------------
# Quantum expectation estimator
# ----------------------------------------------------------------------
class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrised circuit."""
    def __init__(self, circuit: QuantumCircuit):
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: list[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(self, observables: list, parameter_sets: list[list[float]]) -> list[list[complex]]:
        observables = list(observables)
        results: list[list[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

class FastEstimator(FastBaseEstimator):
    """Add Gaussian shot noise to the deterministic estimator."""
    def evaluate(
        self,
        observables: list,
        parameter_sets: list[list[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> list[list[complex]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: list[list[complex]] = []
        for row in raw:
            noisy.append([complex(rng.normal(float(mean), max(1e-6, 1 / shots))) for mean in row])
        return noisy

# ----------------------------------------------------------------------
# Variational quantum gate
# ----------------------------------------------------------------------
class QuantumGate(nn.Module):
    """Variational circuit that maps an angle vector to expectation values."""
    def __init__(self, n_qubits: int, backend: Backend, shots: int = 1024) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots

        # Trainable rotation parameters for each qubit
        self.rz_params = nn.ParameterList([nn.Parameter(torch.randn(1)) for _ in range(n_qubits)])

    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        angles
            Tensor of shape ``(batch, n_qubits)`` containing the rotation
            angles for the RY gate of each qubit.

        Returns
        -------
        torch.Tensor
            Expectation values of Pauli Z for each qubit, shape
            ``(batch, n_qubits)``.
        """
        batch, n = angles.shape
        assert n == self.n_qubits, "angle dimension must match number of qubits"

        out = torch.empty(batch, self.n_qubits, device=angles.device, dtype=torch.float32)

        for i in range(batch):
            circ = QuantumCircuit(self.n_qubits)
            # Encode the angles into RY gates
            for q in range(self.n_qubits):
                circ.ry(angles[i, q].item(), q)
                circ.rz(self.rz_params[q].item(), q)
            circ.measure_all()
            compiled = transpile(circ, self.backend)
            qobj = assemble(compiled, shots=self.shots)
            result = self.backend.run(qobj).result()
            counts = result.get_counts()
            # Compute expectation of Pauli Z for each qubit
            expectation = np.zeros(self.n_qubits, dtype=np.float32)
            for bitstring, cnt in counts.items():
                parity = 1
                for idx, bit in enumerate(reversed(bitstring)):
                    if bit == '1':
                        parity *= -1
                expectation += parity * cnt
            expectation /= self.shots
            out[i] = torch.tensor(expectation, device=angles.device, dtype=torch.float32)
        return out

# ----------------------------------------------------------------------
# Quantum LSTM cell
# ----------------------------------------------------------------------
class QuantumQLSTM(nn.Module):
    """LSTM cell with quantum‑based gates."""
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int,
        backend: Backend,
        shots: int = 1024,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots

        # Linear layers to produce angles for each gate
        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

        # Quantum gates
        self.forget_gate = QuantumGate(n_qubits, backend, shots)
        self.input_gate = QuantumGate(n_qubits, backend, shots)
        self.update_gate = QuantumGate(n_qubits, backend, shots)
        self.output_gate = QuantumGate(n_qubits, backend, shots)

    def forward(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f_angles = self.linear_forget(combined)
            i_angles = self.linear_input(combined)
            g_angles = self.linear_update(combined)
            o_angles = self.linear_output(combined)

            f = torch.sigmoid(self.forget_gate(f_angles))
            i = torch.sigmoid(self.input_gate(i_angles))
            g = torch.tanh(self.update_gate(g_angles))
            o = torch.sigmoid(self.output_gate(o_angles))

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    @staticmethod
    def _init_states(
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, inputs.size(-1), device=device),
            torch.zeros(batch_size, inputs.size(-1), device=device),
        )

# ----------------------------------------------------------------------
# Tagger using the quantum LSTM
# ----------------------------------------------------------------------
class LSTMTagger(nn.Module):
    """
    Sequence tagging model that uses a quantum LSTM cell.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int,
        backend: Backend,
        shots: int = 1024,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QuantumQLSTM(embedding_dim, hidden_dim, n_qubits, backend, shots)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence).transpose(0, 1)  # (seq_len, batch, embed)
        lstm_out, _ = self.lstm(embeds)
        tag_logits = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_logits, dim=-1)

# ----------------------------------------------------------------------
# Compatibility aliases
# ----------------------------------------------------------------------
HybridQLSTM = QuantumQLSTM
QLSTM = QuantumQLSTM
LSTMTagger = LSTMTagger

__all__ = ["HybridQLSTM", "QLSTM", "LSTMTagger", "FastEstimator", "FastBaseEstimator"]
