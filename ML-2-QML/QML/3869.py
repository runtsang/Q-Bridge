"""Quantum evaluator and quantum LSTM implementation using Qiskit.

The module defines:

* :class:`QuantumEvaluator` – evaluates a parametric :class:`~qiskit.circuit.QuantumCircuit`
  for many parameter sets, optionally adding shot noise via the QASM simulator.
* :class:`QuantumQLSTM` – a quantum LSTM cell where each gate is realised
  by a small Qiskit circuit.  The cell operates on a batch of 1‑D vectors
  and returns the hidden state sequence.
* :class:`QuantumLSTMTagger` – sequence‑tagging model that uses the
  quantum LSTM cell.  It mimics the interface of the classical
  :class:`~LSTMTagger` so it can be swapped seamlessly in the hybrid
  estimator.
"""

from __future__ import annotations

import numpy as np
from collections.abc import Iterable, Sequence
from typing import List, Optional

from qiskit import QuantumCircuit, QuantumRegister, Aer, execute, Parameter
from qiskit.quantum_info import Statevector, Pauli
from qiskit.quantum_info.operators import Pauli as PauliOperator
import torch
from torch import nn

# --------------------------------------------------------------------------- #
# Quantum evaluator
# --------------------------------------------------------------------------- #
class QuantumEvaluator:
    """Evaluate a parametric QuantumCircuit for many sets of parameters.

    Parameters
    ----------
    base_circuit:
        The circuit to evaluate.  It must contain parameters that will be
        bound to the values supplied in ``parameter_sets``.
    variational_circuit:
        Optional circuit that will be composed after the base circuit.
    backend:
        Either ``"statevector"`` or ``"qasm"``.  The latter adds shot noise
        via the QASM simulator.
    shots:
        Number of shots to use when the backend is ``"qasm"``.
    """

    def __init__(
        self,
        base_circuit: QuantumCircuit,
        *,
        variational_circuit: Optional[QuantumCircuit] = None,
        backend: str = "statevector",
        shots: Optional[int] = None,
    ) -> None:
        self.base_circuit = base_circuit
        self.variational_circuit = variational_circuit
        self.backend_name = backend
        self.shots = shots
        self._backend = Aer.get_backend(backend)
        self._n_params = len(base_circuit.parameters)

        if variational_circuit:
            # Append variational circuit to base
            self.base_circuit.compose(variational_circuit, inplace=True)
            self._n_params += len(variational_circuit.parameters)

    def _bind(self, param_values: Sequence[float]) -> QuantumCircuit:
        if len(param_values)!= self._n_params:
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = {p: v for p, v in zip(self.base_circuit.parameters, param_values)}
        return self.base_circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[Pauli],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []
        for params in parameter_sets:
            circuit = self._bind(params)
            if self.shots is None:
                # Statevector simulation
                state = Statevector.from_instruction(circuit)
                row = [state.expectation_value(obs) for obs in observables]
            else:
                # QASM shot‑noise simulation
                qc = circuit.copy()
                qc.measure_all()
                job = execute(qc, self._backend, shots=self.shots)
                counts = job.result().get_counts()
                probs = {k: v / self.shots for k, v in counts.items()}
                row = []
                for obs in observables:
                    exp_val = 0.0
                    for bitstring, p in probs.items():
                        # Map bitstring to eigenvalue of the Pauli operator
                        val = 1.0
                        for i, qubit in enumerate(reversed(bitstring)):
                            if qubit == "1":
                                val *= -1.0
                        exp_val += val * p
                    row.append(complex(exp_val))
            results.append(row)
        return results

# --------------------------------------------------------------------------- #
# Quantum LSTM cell
# --------------------------------------------------------------------------- #
class QuantumQLSTM(nn.Module):
    """Quantum LSTM cell where each gate is implemented by a small Qiskit
    parametric circuit.  The cell operates on a batch of 1‑D vectors.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int,
        backend: str = "statevector",
        shots: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots

        # Build parameterised gate circuits
        self.forget_circ = self._build_gate_circuit("forget")
        self.input_circ = self._build_gate_circuit("input")
        self.update_circ = self._build_gate_circuit("update")
        self.output_circ = self._build_gate_circuit("output")

        # Evaluators for each gate
        self.eval_forget = QuantumEvaluator(self.forget_circ, backend=backend, shots=shots)
        self.eval_input = QuantumEvaluator(self.input_circ, backend=backend, shots=shots)
        self.eval_update = QuantumEvaluator(self.update_circ, backend=backend, shots=shots)
        self.eval_output = QuantumEvaluator(self.output_circ, backend=backend, shots=shots)

    def _build_gate_circuit(self, gate_name: str) -> QuantumCircuit:
        """Create a parametric circuit that encodes a vector into rotations."""
        qreg = QuantumRegister(self.n_qubits)
        circuit = QuantumCircuit(qreg)
        params = [Parameter(f"{gate_name}_{i}") for i in range(self.n_qubits)]
        for i, p in enumerate(params):
            circuit.rx(p, qreg[i])
        # Entangle with a simple CNOT chain
        for i in range(self.n_qubits - 1):
            circuit.cx(qreg[i], qreg[i + 1])
        return circuit

    def _quantum_gate(
        self,
        evaluator: QuantumEvaluator,
        vector: np.ndarray,
    ) -> torch.Tensor:
        """Evaluate a gate circuit on a classical vector and return a scalar."""
        # Pad or truncate to the number of parameters
        if len(vector) < self.n_qubits:
            params = np.concatenate([vector, np.zeros(self.n_qubits - len(vector))])
        else:
            params = vector[: self.n_qubits]
        # Use a single Pauli‑Z observable on the first qubit for simplicity
        obs = Pauli("Z" + "I" * (self.n_qubits - 1))
        result = evaluator.evaluate([obs], [params])[0][0]
        return torch.tensor(result.real, dtype=torch.float32)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass over a batch of input vectors."""
        if states is None:
            batch_size = inputs.shape[0]
            hx = torch.zeros(batch_size, self.hidden_dim)
            cx = torch.zeros(batch_size, self.hidden_dim)
        else:
            hx, cx = states

        outputs = []
        for idx in range(inputs.shape[0]):
            x = inputs[idx]
            combined = torch.cat([x, hx[idx]], dim=0).numpy()

            f = torch.sigmoid(self._quantum_gate(self.eval_forget, combined))
            i = torch.sigmoid(self._quantum_gate(self.eval_input, combined))
            g = torch.tanh(self._quantum_gate(self.eval_update, combined))
            o = torch.sigmoid(self._quantum_gate(self.eval_output, combined))

            cx = f * cx[idx] + i * g
            hx[idx] = o * torch.tanh(cx)
            outputs.append(hx[idx].unsqueeze(0))

        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

# --------------------------------------------------------------------------- #
# Quantum LSTM tagger
# --------------------------------------------------------------------------- #
class QuantumLSTMTagger(nn.Module):
    """Sequence‑tagging model that uses the quantum LSTM cell."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QuantumQLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return torch.log_softmax(tag_logits, dim=1)

__all__ = [
    "QuantumEvaluator",
    "QuantumQLSTM",
    "QuantumLSTMTagger",
]
