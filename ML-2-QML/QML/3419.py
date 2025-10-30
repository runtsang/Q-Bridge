"""Hybrid QCNN‑QLSTM architecture for quantum experiments.

The module defines :class:`QCNNQLSTMHybrid` that combines a quantum
QCNN feature extractor with a quantum‑enhanced LSTM gate network.  The
quantum QCNN is built using a variational ansatz that mirrors the
classical convolution‑pooling hierarchy.  The quantum LSTM implements
the gates as small parameter‑efficient circuits based on torchquantum.
The public API matches the classical module so that the class can be
imported interchangeably.
"""

from __future__ import annotations

import torch
from torch import nn
from typing import Tuple

# Quantum circuit libraries
import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals

# Quantum LSTM imports
import torchquantum as tq
import torchquantum.functional as tqf

# --------------------------------------------------------------------------- #
# Quantum QCNN builder
# --------------------------------------------------------------------------- #
def QCNN() -> EstimatorQNN:
    """Constructs the quantum QCNN used in the hybrid class.

    The circuit follows the same convolution‑pooling structure as the
    classical QCNN: 8 qubits, three convolutional layers and three
    pooling layers.  A ZFeatureMap of depth 1 is used for feature
    encoding.  The ansatz is decomposed before being wrapped by an
    :class:`EstimatorQNN` so that it can be treated as a PyTorch
    module.
    """
    algorithm_globals.random_seed = 12345
    estimator = StatevectorEstimator()

    # Convolution circuit
    def conv_circuit(params):
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        qc.cx(1, 0)
        qc.rz(np.pi / 2, 0)
        return qc

    # Convolutional layer
    def conv_layer(num_qubits, param_prefix):
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(param_prefix, length=num_qubits * 3)
        for i in range(0, num_qubits, 2):
            qc.append(conv_circuit(params[i*3:(i+1)*3]), [i, i+1])
            qc.barrier()
        for i in range(1, num_qubits-1, 2):
            if i+1 < num_qubits:
                qc.append(conv_circuit(params[i*3:(i+1)*3]), [i, i+1])
                qc.barrier()
        return qc

    # Pooling circuit
    def pool_circuit(params):
        qc = QuantumCircuit(2)
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(params[0], 0)
        qc.ry(params[1], 1)
        qc.cx(0, 1)
        qc.ry(params[2], 1)
        return qc

    # Pooling layer
    def pool_layer(sources, sinks, param_prefix):
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(param_prefix, length=(len(sources)//2) * 3)
        idx = 0
        for src, sink in zip(sources, sinks):
            qc.append(pool_circuit(params[idx:idx+3]), [src, sink])
            qc.barrier()
            idx += 3
        return qc

    # Assemble the ansatz
    ansatz = QuantumCircuit(8)
    ansatz.compose(conv_layer(8, "c1"), inplace=True)
    ansatz.compose(pool_layer([0,1,2,3], [4,5,6,7], "p1"), inplace=True)
    ansatz.compose(conv_layer(4, "c2"), inplace=True)
    ansatz.compose(pool_layer([0,1], [2,3], "p2"), inplace=True)
    ansatz.compose(conv_layer(2, "c3"), inplace=True)
    ansatz.compose(pool_layer([0], [1], "p3"), inplace=True)

    # Combine feature map and ansatz
    feature_map = ZFeatureMap(8)
    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, inplace=True)
    circuit.compose(ansatz, inplace=True)

    # Observable for the output
    observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])

    # Wrap into EstimatorQNN
    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )
    return qnn

# --------------------------------------------------------------------------- #
# Quantum LSTM implementation (from seed)
# --------------------------------------------------------------------------- #
class _QLSTMGate(tq.QuantumModule):
    """Gate realized by a small quantum circuit."""
    def __init__(self, n_wires: int) -> None:
        super().__init__()
        self.n_wires = n_wires
        # Simple encoder: RX rotations on each wire
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
        )
        # Parameterized RX gates
        self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
        self.encoder(qdev, x)
        for gate, wire in zip(self.params, range(self.n_wires)):
            gate(qdev, wires=wire)
        for wire in range(self.n_wires - 1):
            tqf.cnot(qdev, wires=[wire, wire + 1])
        tqf.cnot(qdev, wires=[self.n_wires - 1, 0])
        return self.measure(qdev)

class _QuantumQLSTM(nn.Module):
    """Quantum LSTM cell using the gates defined above."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.forget = _QLSTMGate(n_qubits)
        self.input = _QLSTMGate(n_qubits)
        self.update = _QLSTMGate(n_qubits)
        self.output = _QLSTMGate(n_qubits)
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
            f = torch.sigmoid(self.forget(self.linear_forget(combined)))
            i = torch.sigmoid(self.input(self.linear_input(combined)))
            g = torch.tanh(self.update(self.linear_update(combined)))
            o = torch.sigmoid(self.output(self.linear_output(combined)))
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
# Hybrid QCNN‑QLSTM tagger
# --------------------------------------------------------------------------- #
class QCNNQLSTMHybrid(nn.Module):
    """Drop‑in replacement for the original LSTMTagger that uses a quantum
    QCNN feature extractor followed by a quantum LSTM.  The class accepts a
    ``n_qubits`` argument to control the size of the quantum gates.
    """
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
        # Quantum QCNN produces a scalar per batch element
        self.qcnn = QCNN()
        # Quantum LSTM takes the QCNN output (scalar) as input
        self.lstm = _QuantumQLSTM(1, hidden_dim, n_qubits)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that runs the quantum QCNN and quantum LSTM.

        Parameters
        ----------
        sentence : torch.Tensor
            Tensor of token indices with shape ``(seq_len, batch)``.
        """
        embeds = self.word_embeddings(sentence)  # (seq_len, batch, embedding_dim)
        # Apply QCNN to each time step
        qcnn_features = torch.stack([self.qcnn(e) for e in embeds], dim=0)
        # qcnn_features shape: (seq_len, batch, 1)
        lstm_out, _ = self.lstm(qcnn_features)
        tag_logits = self.hidden2tag(lstm_out)
        return torch.log_softmax(tag_logits, dim=-1)

__all__ = ["QCNNQLSTMHybrid"]
