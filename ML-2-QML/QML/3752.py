import torch
import torch.nn as nn
import torch.quantum as tq
import torch.quantum.functional as tqf
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators import BaseOperator
import numpy as np
from typing import Iterable, Sequence, List, Tuple, Callable

# --------------------------------------------------------------------------- #
# Quantum LSTM with richer variational gates
# --------------------------------------------------------------------------- #
class QLSTMQuantum(tq.QuantumModule):
    """
    Variational LSTM where each gate is realised by a small
    parameterised quantum circuit followed by classical post‑processing.
    """
    class _GateLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
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
            for gate in self.params:
                gate(qdev)
            for wire in range(self.n_wires - 1):
                tqf.cnot(qdev, wires=[wire, wire + 1])
            tqf.cnot(qdev, wires=[self.n_wires - 1, 0])
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        self.forget_gate = self._GateLayer(n_qubits)
        self.input_gate = self._GateLayer(n_qubits)
        self.update_gate = self._GateLayer(n_qubits)
        self.output_gate = self._GateLayer(n_qubits)

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
            f = torch.sigmoid(self.forget_gate(self.linear_forget(combined)))
            i = torch.sigmoid(self.input_gate(self.linear_input(combined)))
            g = torch.tanh(self.update_gate(self.linear_update(combined)))
            o = torch.sigmoid(self.output_gate(self.linear_output(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        return torch.cat(outputs, dim=0), (hx, cx)

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

# --------------------------------------------------------------------------- #
# Fast quantum estimator using qiskit Statevector
# --------------------------------------------------------------------------- #
class FastQuantumEstimator:
    """
    Evaluates expectation values of qiskit BaseOperator observables for a
    parameterised circuit. Supports optional shot‑noise simulation.
    """
    def __init__(self, circuit: QuantumCircuit):
        self._circuit = circuit
        self._params = list(circuit.parameters)

    def _bind(self, param_values: Sequence[float]) -> QuantumCircuit:
        if len(param_values)!= len(self._params):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._params, param_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []
        rng = np.random.default_rng(seed)
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            if shots is not None:
                noise = rng.normal(0, 1 / np.sqrt(shots), size=len(row))
                row = [v + n for v, n in zip(row, noise)]
            results.append(row)
        return results

# --------------------------------------------------------------------------- #
# Quantum LSTMTagger with quantum estimator
# --------------------------------------------------------------------------- #
class QLSTMHybridQuantum(nn.Module):
    """
    Quantum‑enhanced LSTMTagger that exposes a fast estimator
    based on qiskit Statevector evaluations.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int,
        circuit: QuantumCircuit,
    ) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QLSTMQuantum(embedding_dim, hidden_dim, n_qubits)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.estimator = FastQuantumEstimator(circuit)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return torch.log_softmax(tag_logits, dim=1)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        return self.estimator.evaluate(observables, parameter_sets, shots=shots, seed=seed)

__all__ = ["QLSTMQuantum", "FastQuantumEstimator", "QLSTMHybridQuantum"]
