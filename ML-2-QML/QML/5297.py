"""Hybrid quantum LSTM implementation with auxiliary utilities.

This module mirrors the classical version but replaces all linear
gates with small variational quantum circuits.  The public API
remains identical to the original `QLSTM` so that it can be used
as a drop‑in replacement.

The implementation is built on top of torchquantum and
qiskit for the estimator, and adds shot‑noise support in
`FastEstimator`.  A simple quantum regression network
(`QModel`) and dataset are also provided.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from collections.abc import Iterable, Sequence
from typing import Callable, List, Tuple, Optional

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class QLayer(tq.QuantumModule):
    """Variational layer that implements a small quantum circuit.

    The circuit consists of an initial general encoder followed
    by a trainable RX–RY layer and a CNOT chain that entangles
    the wires.  The output of the layer is a measurement in the
    Pauli‑Z basis, which is fed into the LSTM gate.
    """

    def __init__(self, n_wires: int) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
        )
        self.params = nn.ModuleList(
            [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
        # The input vector must already be loaded into qdev.x
        self.encoder(qdev, qdev.x)
        for wire, gate in enumerate(self.params):
            gate(qdev, wires=wire)
        # Entangle wires with a linear chain of CNOTs
        for wire in range(self.n_wires - 1):
            tqf.cnot(qdev, wires=[wire, wire + 1])
        return self.measure(qdev)


class QLSTM(nn.Module):
    """Quantum‑enhanced LSTM cell where each gate is a variational
    quantum circuit.  The linear pre‑processing maps the classical
    concatenated input into a register of qubits that is then
    processed by `QLayer`.  The final measurement is used as the
    gate activation.
    """

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Variational quantum gates
        self.forget = QLayer(n_qubits)
        self.input = QLayer(n_qubits)
        self.update = QLayer(n_qubits)
        self.output = QLayer(n_qubits)

        # Linear layers that map the classical concatenated vector
        # into the qubit register
        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            # Classical linear mapping to qubit register
            f_in = self.linear_forget(combined)
            i_in = self.linear_input(combined)
            g_in = self.linear_update(combined)
            o_in = self.linear_output(combined)

            # Quantum gates
            qdev_f = tq.QuantumDevice(n_wires=self.n_qubits, bsz=f_in.shape[0], device=f_in.device)
            qdev_f.x = f_in
            f = torch.sigmoid(self.forget(qdev_f))

            qdev_i = tq.QuantumDevice(n_wires=self.n_qubits, bsz=i_in.shape[0], device=i_in.device)
            qdev_i.x = i_in
            i = torch.sigmoid(self.input(qdev_i))

            qdev_g = tq.QuantumDevice(n_wires=self.n_qubits, bsz=g_in.shape[0], device=g_in.device)
            qdev_g.x = g_in
            g = torch.tanh(self.update(qdev_g))

            qdev_o = tq.QuantumDevice(n_wires=self.n_qubits, bsz=o_in.shape[0], device=o_in.device)
            qdev_o.x = o_in
            o = torch.sigmoid(self.output(qdev_o))

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)


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
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits) if n_qubits > 0 else nn.LSTM(
            embedding_dim, hidden_dim
        )

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        return F.log_softmax(self.hidden2tag(lstm_out.view(len(sentence), -1)), dim=1)


class FastBaseEstimator:
    """Evaluate a quantum circuit for batches of parameters and observables."""

    def __init__(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(self, observables: Iterable[BaseOperator], parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            results.append([state.expectation_value(obs) for obs in observables])
        return results


class FastEstimator(FastBaseEstimator):
    """Adds Gaussian shot noise to the deterministic estimator."""

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in raw:
            noisy_row = [
                rng.normal(val.real, max(1e-6, 1 / shots)) + 1j * rng.normal(val.imag, max(1e-6, 1 / shots))
                for val in row
            ]
            noisy.append(noisy_row)
        return noisy


class EstimatorQNN:
    """Quantum circuit that implements a simple variational circuit
    for regression.  The circuit is built with qiskit and evaluated
    with the StatevectorEstimator.
    """

    def __init__(self) -> None:
        # Parameters for input and weight
        self.input_param = Parameter("input1")
        self.weight_param = Parameter("weight1")
        self.circuit = QuantumCircuit(1)
        self.circuit.h(0)
        self.circuit.ry(self.input_param, 0)
        self.circuit.rx(self.weight_param, 0)
        # Observable
        self.observable = BaseOperator("Y")

    def evaluate(self, inputs: torch.Tensor) -> torch.Tensor:
        """Evaluate the circuit for a batch of inputs."""
        from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
        from qiskit.primitives import StatevectorEstimator
        estimator = StatevectorEstimator()
        estimator_qnn = QiskitEstimatorQNN(
            circuit=self.circuit,
            observables=self.observable,
            input_params=[self.input_param],
            weight_params=[self.weight_param],
            estimator=estimator,
        )
        return estimator_qnn(inputs)


def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate quantum superposition states and labels."""
    omega_0 = np.zeros(2 ** num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_wires, dtype=complex)
    omega_1[-1] = 1.0

    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels.astype(np.float32)


class RegressionDataset(torch.utils.data.Dataset):
    """Dataset for quantum regression."""

    def __init__(self, samples: int, num_wires: int) -> None:
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class QModel(tq.QuantumModule):
    """Quantum regression network."""

    class QLayer(tq.QuantumModule):
        def __init__(self, num_wires: int):
            super().__init__()
            self.n_wires = num_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random_layer(qdev)
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)

    def __init__(self, num_wires: int) -> None:
        super().__init__()
        self.n_wires = num_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self.QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)


__all__ = [
    "QLSTM",
    "LSTMTagger",
    "FastBaseEstimator",
    "FastEstimator",
    "EstimatorQNN",
    "RegressionDataset",
    "QModel",
    "generate_superposition_data",
]
