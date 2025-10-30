from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Tuple, Union

import numpy as np
import torch
from torch import nn

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator

import qutip as qt
import torchquantum as tq
import torchquantum.functional as tqf

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


class FastBaseEstimator:
    """Quantum‑centric estimator that evaluates Qiskit circuits or torchquantum modules."""

    def __init__(self, model: Union[QuantumCircuit, tq.QuantumModule], *, n_qubits: int | None = None) -> None:
        self.model = model
        self.n_qubits = n_qubits

    # ------------------------------------------------------------------
    #  Evaluation dispatch
    # ------------------------------------------------------------------
    def evaluate(
        self,
        observables: Iterable[BaseOperator | torch.Tensor],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex | torch.Tensor]]:
        """Dispatch evaluation to the appropriate backend."""
        if isinstance(self.model, QuantumCircuit):
            return self._evaluate_circuit(observables, parameter_sets)  # type: ignore[arg-type]
        elif isinstance(self.model, tq.QuantumModule):
            return self._evaluate_tq(observables, parameter_sets)  # type: ignore[arg-type]
        else:
            raise TypeError("Unsupported model type: %s" % type(self.model))

    # ------------------------------------------------------------------
    #  Qiskit circuit evaluation
    # ------------------------------------------------------------------
    def _bind_circuit(self, param_vals: Sequence[float]) -> QuantumCircuit:
        if len(param_vals)!= len(self.model.parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self.model.parameters, param_vals))
        return self.model.assign_parameters(mapping, inplace=False)

    def _evaluate_circuit(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind_circuit(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

    # ------------------------------------------------------------------
    #  torchquantum evaluation
    # ------------------------------------------------------------------
    def _evaluate_tq(
        self,
        observables: Iterable[torch.Tensor],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[torch.Tensor]]:
        results: List[List[torch.Tensor]] = []
        for params in parameter_sets:
            inputs = torch.as_tensor(params, dtype=torch.float32).unsqueeze(0)
            outputs = self.model(inputs)
            row = [obs(outputs) for obs in observables]
            results.append(row)
        return results

    # ------------------------------------------------------------------
    #  Graph‑QNN utilities (quantum)
    # ------------------------------------------------------------------
    @staticmethod
    def _tensored_id(num_qubits: int) -> qt.Qobj:
        identity = qt.qeye(2 ** num_qubits)
        dims = [2] * num_qubits
        identity.dims = [dims.copy(), dims.copy()]
        return identity

    @staticmethod
    def _tensored_zero(num_qubits: int) -> qt.Qobj:
        projector = qt.fock(2 ** num_qubits).proj()
        dims = [2] * num_qubits
        projector.dims = [dims.copy(), dims.copy()]
        return projector

    @staticmethod
    def _swap_registers(op: qt.Qobj, source: int, target: int) -> qt.Qobj:
        if source == target:
            return op
        order = list(range(len(op.dims[0])))
        order[source], order[target] = order[target], order[source]
        return op.permute(order)

    @staticmethod
    def _random_qubit_unitary(num_qubits: int) -> qt.Qobj:
        dim = 2 ** num_qubits
        matrix = np.random.normal(size=(dim, dim)) + 1j * np.random.normal(size=(dim, dim))
        unitary = np.linalg.orth(matrix)
        qobj = qt.Qobj(unitary)
        dims = [2] * num_qubits
        qobj.dims = [dims.copy(), dims.copy()]
        return qobj

    @staticmethod
    def _random_qubit_state(num_qubits: int) -> qt.Qobj:
        dim = 2 ** num_qubits
        amplitudes = np.random.normal(size=(dim, 1)) + 1j * np.random.normal(size=(dim, 1))
        amplitudes /= np.linalg.norm(amplitudes)
        state = qt.Qobj(amplitudes)
        state.dims = [[2] * num_qubits, [1] * num_qubits]
        return state

    @staticmethod
    def random_network(
        qnn_arch: list[int], samples: int
    ) -> Tuple[list[int], list[list[qt.Qobj]], list[tuple[qt.Qobj, qt.Qobj]], qt.Qobj]:
        """Construct a random parameterized quantum network."""
        target_unitary = FastBaseEstimator._random_qubit_unitary(qnn_arch[-1])
        training_data = FastBaseEstimator.random_training_data(target_unitary, samples)

        unitaries: list[list[qt.Qobj]] = [[]]
        for layer in range(1, len(qnn_arch)):
            num_inputs = qnn_arch[layer - 1]
            num_outputs = qnn_arch[layer]
            layer_ops: list[qt.Qobj] = []
            for output in range(num_outputs):
                op = FastBaseEstimator._random_qubit_unitary(num_inputs + 1)
                if num_outputs > 1:
                    op = qt.tensor(FastBaseEstimator._random_qubit_unitary(num_inputs + 1), FastBaseEstimator._tensored_id(num_outputs - 1))
                    op = FastBaseEstimator._swap_registers(op, num_inputs, num_inputs + output)
                layer_ops.append(op)
            unitaries.append(layer_ops)

        return qnn_arch, unitaries, training_data, target_unitary

    @staticmethod
    def random_training_data(unitary: qt.Qobj, samples: int) -> list[tuple[qt.Qobj, qt.Qobj]]:
        dataset = []
        num_qubits = len(unitary.dims[0])
        for _ in range(samples):
            state = FastBaseEstimator._random_qubit_state(num_qubits)
            dataset.append((state, unitary * state))
        return dataset

    @staticmethod
    def _partial_trace_keep(state: qt.Qobj, keep: Sequence[int]) -> qt.Qobj:
        if len(keep)!= len(state.dims[0]):
            return state.ptrace(list(keep))
        return state

    @staticmethod
    def _partial_trace_remove(state: qt.Qobj, remove: Sequence[int]) -> qt.Qobj:
        keep = list(range(len(state.dims[0])))
        for index in sorted(remove, reverse=True):
            keep.pop(index)
        return FastBaseEstimator._partial_trace_keep(state, keep)

    @staticmethod
    def _layer_channel(
        qnn_arch: Sequence[int],
        unitaries: Sequence[Sequence[qt.Qobj]],
        layer: int,
        input_state: qt.Qobj,
    ) -> qt.Qobj:
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        state = qt.tensor(input_state, FastBaseEstimator._tensored_zero(num_outputs))

        layer_unitary = unitaries[layer][0].copy()
        for gate in unitaries[layer][1:]:
            layer_unitary = gate * layer_unitary

        return FastBaseEstimator._partial_trace_remove(
            layer_unitary * state * layer_unitary.dag(), range(num_inputs)
        )

    @staticmethod
    def feedforward(
        qnn_arch: Sequence[int],
        unitaries: Sequence[Sequence[qt.Qobj]],
        samples: Iterable[tuple[qt.Qobj, qt.Qobj]],
    ):
        """Forward propagate through the quantum network."""
        stored_states = []
        for sample, _ in samples:
            layerwise = [sample]
            current_state = sample
            for layer in range(1, len(qnn_arch)):
                current_state = FastBaseEstimator._layer_channel(qnn_arch, unitaries, layer, current_state)
                layerwise.append(current_state)
            stored_states.append(layerwise)
        return stored_states

    @staticmethod
    def state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
        """Return the absolute squared overlap between pure states ``a`` and ``b``."""
        return abs((a.dag() * b)[0, 0]) ** 2

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[qt.Qobj],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Create a weighted adjacency graph from state fidelities."""
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = FastBaseEstimator.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    # ------------------------------------------------------------------
    #  Quantum LSTM tagger
    # ------------------------------------------------------------------
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

        def forward(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] = None):
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
        """Sequence tagging model that can switch between classical and quantum LSTM."""

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
                self.lstm = FastBaseEstimator.QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
            else:
                self.lstm = nn.LSTM(embedding_dim, hidden_dim)
            self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

        def forward(self, sentence: torch.Tensor) -> torch.Tensor:
            embeds = self.word_embeddings(sentence)
            lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
            tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
            return torch.nn.functional.log_softmax(tag_logits, dim=1)


__all__ = [
    "FastBaseEstimator",
]
