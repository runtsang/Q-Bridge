"""Hybrid LSTM with quantum gates and quantum QCNN feature extractor.

This module implements the quantum counterpart of :class:`HybridQLSTM`.
The interface mirrors the classical version so that the same script can
switch between regimes by importing the appropriate module.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
import qiskit.quantum_info as qi
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
import numpy as np
import torchquantum as tq
import torchquantum.functional as tqf

# Quantum LSTM cell – gates realised by small variational circuits.
class QuantumQLSTM(nn.Module):
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
            for gate in self.params:
                gate(qdev)
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

    def forward(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
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
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

# Quantum QCNN feature extractor
class QuantumQCNN(nn.Module):
    def __init__(self, input_dim: int = 8) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.estimator = Estimator()
        self.circuit = self._build_qcnn()
        weight_params = [p for p in self.circuit.parameters if "c" in p.name]
        self.qnn = EstimatorQNN(
            circuit=self.circuit.decompose(),
            observables=qi.SparsePauliOp.from_list([("Z" + "I" * 7, 1)]),
            input_params=self.circuit.parameters,
            weight_params=weight_params,
            estimator=self.estimator,
        )

    def _build_qcnn(self) -> qiskit.QuantumCircuit:
        def conv_circuit(params):
            qc = qiskit.QuantumCircuit(2)
            qc.rz(-np.pi / 2, 1)
            qc.cx(1, 0)
            qc.rz(params[0], 0)
            qc.ry(params[1], 1)
            qc.cx(0, 1)
            qc.ry(params[2], 1)
            qc.cx(1, 0)
            qc.rz(np.pi / 2, 0)
            return qc

        def conv_layer(num_qubits, param_prefix):
            qc = qiskit.QuantumCircuit(num_qubits, name="Convolutional Layer")
            qubits = list(range(num_qubits))
            param_index = 0
            params = ParameterVector(param_prefix, length=num_qubits * 3)
            for q1, q2 in zip(qubits[0::2], qubits[1::2]):
                qc = qc.compose(conv_circuit(params[param_index : param_index + 3]), [q1, q2])
                qc.barrier()
                param_index += 3
            for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
                qc = qc.compose(conv_circuit(params[param_index : param_index + 3]), [q1, q2])
                qc.barrier()
                param_index += 3
            return qc

        def pool_circuit(params):
            qc = qiskit.QuantumCircuit(2)
            qc.rz(-np.pi / 2, 1)
            qc.cx(1, 0)
            qc.rz(params[0], 0)
            qc.ry(params[1], 1)
            qc.cx(0, 1)
            qc.ry(params[2], 1)
            return qc

        def pool_layer(sources, sinks, param_prefix):
            num_qubits = len(sources) + len(sinks)
            qc = qiskit.QuantumCircuit(num_qubits, name="Pooling Layer")
            param_index = 0
            params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
            for source, sink in zip(sources, sinks):
                qc = qc.compose(pool_circuit(params[param_index : param_index + 3]), [source, sink])
                qc.barrier()
                param_index += 3
            return qc

        ansatz = qiskit.QuantumCircuit(8, name="Ansatz")
        ansatz.compose(conv_layer(8, "c1"), list(range(8)), inplace=True)
        ansatz.compose(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), list(range(8)), inplace=True)
        ansatz.compose(conv_layer(4, "c2"), list(range(4, 8)), inplace=True)
        ansatz.compose(pool_layer([0, 1], [2, 3], "p2"), list(range(4, 8)), inplace=True)
        ansatz.compose(conv_layer(2, "c3"), list(range(6, 8)), inplace=True)
        ansatz.compose(pool_layer([0], [1], "p3"), list(range(6, 8)), inplace=True)

        feature_map = ZFeatureMap(8)
        circuit = qiskit.QuantumCircuit(8)
        circuit.compose(feature_map, range(8), inplace=True)
        circuit.compose(ansatz, range(8), inplace=True)
        return circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 8)
        x_np = x.detach().cpu().numpy()
        input_dicts = [{p: val for p, val in zip(self.circuit.parameters, vec)} for vec in x_np]
        outputs = self.qnn.predict(input_dicts)
        return torch.tensor(outputs, device=x.device, dtype=x.dtype).unsqueeze(-1)

# Unified hybrid model with quantum back‑ends
class HybridQLSTM(nn.Module):
    """Drop‑in replacement that can switch between classical and quantum back‑ends.

    Parameters
    ----------
    embedding_dim : int
        Dimension of the input embeddings.
    hidden_dim : int
        Hidden size of the LSTM layer.
    vocab_size : int
        Size of the vocabulary.
    tagset_size : int
        Number of distinct tags.
    use_qcircuit : bool, optional
        If ``True`` the LSTM gates are realised by quantum circuits.
    use_qconv : bool, optional
        If ``True`` the input embeddings are passed through a quantum
        QCNN‑style feature extractor before the LSTM.
    n_qubits : int, optional
        Number of qubits used by the quantum gates.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        use_qcircuit: bool = False,
        use_qconv: bool = False,
        n_qubits: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.use_qconv = use_qconv
        self.use_qcircuit = use_qcircuit
        if use_qconv:
            self.qconv = QuantumQCNN()
        else:
            self.qconv = None
        if use_qcircuit:
            self.lstm = QuantumQLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.embedding(sentence)
        if self.use_qconv:
            embeds = self.qconv(embeds)
        lstm_out, _ = self.lstm(embeds.unsqueeze(1))
        logits = self.hidden2tag(lstm_out.squeeze(1))
        return F.log_softmax(logits, dim=1)

__all__ = ["HybridQLSTM"]
