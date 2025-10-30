"""Hybrid LSTM model with quantum feature extractor, quantum LSTM cell, and quantum fully connected layer.

The architecture mirrors the classical version but replaces the optional QCNN, LSTM, and classification
layers with their quantum counterparts when `n_qubits > 0`.  The module is fully importable and
provides the same API as the classical variant.

Key quantum primitives:
- `QLayer` implements a parameterised quantum gate block using TorchQuantum.
- `QLSTMQuantum` is a quantum‑enhanced LSTM cell.
- `QuantumQCNN` builds a QCNN circuit using Qiskit, returning an EstimatorQNN.
- `QuantumFCL` is a parameterised single‑qubit measurement that acts as a fully connected layer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Quantum libraries
import torchquantum as tq
import torchquantum.functional as tqf
import qiskit
import numpy as np
from qiskit import Aer
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator


# Quantum fully connected layer
class QuantumFCL:
    """
    Simple parameterised single‑qubit circuit that returns the expectation value of `Z`
    after applying a rotation around `Y` by the input parameters.
    """
    def __init__(self, n_qubits: int = 1, shots: int = 1024):
        self.backend = Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        self.theta = qiskit.circuit.Parameter("theta")
        self.circuit.h(range(n_qubits))
        self.circuit.ry(self.theta, range(n_qubits))
        self.circuit.measure_all()

    def run(self, thetas):
        job = qiskit.execute(
            self.circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        result = job.result().get_counts(self.circuit)
        counts = np.array(list(result.values()))
        states = np.array(list(result.keys())).astype(float)
        probabilities = counts / self.shots
        expectation = np.sum(states * probabilities)
        return np.array([expectation])


# Quantum QCNN layer
def QuantumQCNN(num_qubits: int = 8):
    """
    Builds a QCNN circuit that mimics the structure from the reference implementation.
    Returns an EstimatorQNN that can be used as a differentiable layer.
    """
    # Feature map
    feature_map = ZFeatureMap(num_qubits)

    # Convolutional unitary
    def conv_circuit(params):
        target = qiskit.QuantumCircuit(num_qubits)
        target.rz(-np.pi / 2, 0)
        target.cx(0, 1)
        target.rz(params[0], 0)
        target.ry(params[1], 1)
        target.cx(1, 0)
        target.ry(params[2], 1)
        target.cx(0, 1)
        target.rz(np.pi / 2, 0)
        return target

    # Pooling unitary
    def pool_circuit(params):
        target = qiskit.QuantumCircuit(num_qubits)
        target.rz(-np.pi / 2, 0)
        target.cx(0, 1)
        target.rz(params[0], 0)
        target.ry(params[1], 1)
        target.cx(1, 0)
        target.ry(params[2], 1)
        return target

    # Compose layers (simplified for brevity)
    qc = qiskit.QuantumCircuit(num_qubits)
    qc.append(feature_map, range(num_qubits))

    conv_params = ParameterVector("c", length=num_qubits * 3)
    pool_params = ParameterVector("p", length=num_qubits * 3)
    for i in range(num_qubits):
        qc.append(conv_circuit(conv_params[i*3:(i+1)*3]), [i, (i+1)%num_qubits])
        qc.append(pool_circuit(pool_params[i*3:(i+1)*3]), [i, (i+1)%num_qubits])

    observable = SparsePauliOp.from_list([("Z" + "I" * (num_qubits - 1), 1)])

    estimator = StatevectorEstimator()
    return EstimatorQNN(
        circuit=qc,
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=conv_params + pool_params,
        estimator=estimator,
    )


# Quantum LSTM cell
class QLayer(tq.QuantumModule):
    """
    Tiny quantum gate block that applies parameterised rotations and a chain of CNOTs.
    """
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
            target = (wire + 1) % self.n_wires
            tqf.cnot(qdev, wires=[wire, target])
        return self.measure(qdev)


class QLSTMQuantum(nn.Module):
    """
    Quantum‑enhanced LSTM cell that uses `QLayer` for each gate.
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        self.forget = QLayer(n_qubits)
        self.input = QLayer(n_qubits)
        self.update = QLayer(n_qubits)
        self.output = QLayer(n_qubits)

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
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

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


# Hybrid model
class HybridQLSTM(nn.Module):
    """
    Hybrid LSTM for sequence tagging that optionally uses a quantum QCNN,
    a quantum LSTM cell, and a quantum fully‑connected layer.

    Parameters
    ----------
    embedding_dim : int
        Size of the word embeddings.
    hidden_dim : int
        Hidden state size.
    vocab_size : int
        Vocabulary size.
    tagset_size : int
        Number of output tags.
    n_qubits : int, default 0
        If > 0, quantum layers are used.
    use_qcnn : bool, default False
        If True, a QCNN quantum feature extractor is applied before the LSTM.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        use_qcnn: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.use_qcnn = use_qcnn
        self.feature_extractor = QuantumQCNN(num_qubits=n_qubits) if use_qcnn else None

        self.lstm = (
            QLSTMQuantum(embedding_dim, hidden_dim, n_qubits=n_qubits)
            if n_qubits > 0
            else nn.LSTM(embedding_dim, hidden_dim)
        )

        self.fcl = QuantumFCL(n_qubits=1)
        self.tagset_size = tagset_size

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a batch of sequences.

        Parameters
        ----------
        sentence : torch.Tensor
            Tensor of token indices of shape (seq_len, batch).

        Returns
        -------
        torch.Tensor
            Log‑softmax logits of shape (seq_len, batch, tagset_size).
        """
        embeds = self.word_embeddings(sentence)  # (seq_len, batch, embed)

        if self.feature_extractor is not None:
            seq_len, batch, _ = embeds.shape
            flat = embeds.view(seq_len * batch, -1)
            # QCNN expects input parameters; we normalize to [0,1]
            flat_norm = (flat - flat.min()) / (flat.max() - flat.min() + 1e-6)
            features = self.feature_extractor(flat_norm)  # (seq_len*batch, 1)
            embeds = features.view(seq_len, batch, -1)

        lstm_out, _ = self.lstm(embeds)  # (seq_len, batch, hidden)

        logits = []
        for step in lstm_out.split(1, dim=0):
            step = step.squeeze(0)  # (batch, hidden)
            step_logits = []
            for h in step.split(1, dim=0):
                h = h.squeeze(0)  # (hidden,)
                expectation = self.fcl.run(h.tolist())
                step_logits.append(expectation[0])
            logits.append(torch.tensor(step_logits, device=embeds.device))
        logits = torch.stack(logits, dim=0)  # (seq_len, batch)

        # Expand to tagset dimension
        logits = logits.unsqueeze(-1).expand(-1, -1, self.tagset_size)

        return F.log_softmax(logits, dim=-1)


__all__ = ["HybridQLSTM"]
