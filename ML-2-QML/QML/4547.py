"""Hybrid quantum LSTM model with optional quantum convolution and estimator.

This module merges concepts from:
- Quantum‑enhanced LSTM gates (torchquantum)
- Quantum convolutional filter (qiskit)
- Quantum estimator neural network (qiskit_machine_learning)
- Fraud‑detection inspired parameterised layers (classical)
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Quantum libraries
import torchquantum as tq
import torchquantum.functional as tqf
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator

# --------------------------------------------------------------------------- #
# Classical fraud‑detection style layer (kept for hybrid compatibility)
# --------------------------------------------------------------------------- #
class FraudLayer(nn.Module):
    """Parameterised linear layer with optional clipping, activation, scaling and shifting."""
    def __init__(self,
                 bs_theta: float,
                 bs_phi: float,
                 phases: tuple[float, float],
                 squeeze_r: tuple[float, float],
                 squeeze_phi: tuple[float, float],
                 displacement_r: tuple[float, float],
                 displacement_phi: tuple[float, float],
                 kerr: tuple[float, float],
                 clip: bool = False) -> None:
        super().__init__()
        weight = torch.tensor([[bs_theta, bs_phi],
                               [squeeze_r[0], squeeze_r[1]]], dtype=torch.float32)
        bias = torch.tensor(phases, dtype=torch.float32)
        if clip:
            weight = weight.clamp(-5.0, 5.0)
            bias = bias.clamp(-5.0, 5.0)
        self.linear = nn.Linear(2, 2)
        with torch.no_grad():
            self.linear.weight.copy_(weight)
            self.linear.bias.copy_(bias)
        self.activation = nn.Tanh()
        self.register_buffer("scale", torch.tensor(displacement_r, dtype=torch.float32))
        self.register_buffer("shift", torch.tensor(displacement_phi, dtype=torch.float32))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        out = self.activation(self.linear(inputs))
        out = out * self.scale + self.shift
        return out

def build_fraud_detection_program(input_params, layers):
    """Create a sequential model mirroring the photonic fraud‑detection structure."""
    modules = [FraudLayer(**vars(input_params), clip=False)]
    modules += [FraudLayer(**vars(l), clip=True) for l in layers]
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

# --------------------------------------------------------------------------- #
# Quantum LSTM gate module
# --------------------------------------------------------------------------- #
class QLayer(tq.QuantumModule):
    """Quantum gate module used inside the quantum LSTM."""
    def __init__(self, n_wires: int) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
        )
        self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
        self.encoder(qdev, x)
        for wire, gate in enumerate(self.params):
            gate(qdev, wires=wire)
        for wire in range(self.n_wires - 1):
            tqf.cnot(qdev, wires=[wire, wire + 1])
        tqf.cnot(qdev, wires=[self.n_wires - 1, 0])
        return self.measure(qdev)

class QuantumQLSTM(nn.Module):
    """LSTM cell where gates are realised by small quantum circuits."""
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

    def forward(self, inputs: torch.Tensor, states: tuple | None = None):
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

    def _init_states(self, inputs: torch.Tensor, states: tuple | None = None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))

# --------------------------------------------------------------------------- #
# Quantum convolutional filter
# --------------------------------------------------------------------------- #
class QuanvCircuit:
    """Quantum convolution filter used for quanvolution layers."""
    def __init__(self, kernel_size: int, backend=None, shots: int = 100, threshold: float = 0.5):
        self.n_qubits = kernel_size ** 2
        self._circuit = QuantumCircuit(self.n_qubits)
        self.theta = [Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit += random_circuit(self.n_qubits, 2, measure=True)
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.threshold = threshold

    def run(self, data):
        """Run the quantum circuit on classical data.

        Args:
            data: 2D array with shape (kernel_size, kernel_size).

        Returns:
            float: average probability of measuring |1> across qubits.
        """
        data = data.reshape(1, self.n_qubits)
        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = math.pi if val > self.threshold else 0
            param_binds.append(bind)
        job = execute(self._circuit,
                      self.backend,
                      shots=self.shots,
                      parameter_binds=param_binds)
        result = job.result().get_counts(self._circuit)
        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val
        return counts / (self.shots * self.n_qubits)

# --------------------------------------------------------------------------- #
# Quantum estimator wrapper
# --------------------------------------------------------------------------- #
class EstimatorQNNWrapper:
    """Wraps a qiskit EstimatorQNN for regression."""
    def __init__(self):
        qc = QuantumCircuit(1)
        w = Parameter("w")
        qc.h(0)
        qc.ry(w, 0)
        qc.rx(w, 0)
        observable = SparsePauliOp.from_list([("Z", 1)])
        estimator = StatevectorEstimator()
        self.estimator_qnn = EstimatorQNN(
            circuit=qc,
            observables=observable,
            input_params=[w],
            weight_params=[w],
            estimator=estimator,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs shape: (N, hidden_dim)
        # use first feature as input parameter
        inp_vals = inputs.detach().cpu().numpy()[:, 0]
        predictions = []
        for val in inp_vals:
            pred = self.estimator_qnn.predict([[val]])[0]
            predictions.append(pred)
        return torch.tensor(predictions, device=inputs.device, dtype=inputs.dtype)

# --------------------------------------------------------------------------- #
# Hybrid quantum‑classical sequence‑tagger
# --------------------------------------------------------------------------- #
class HybridQLSTM(nn.Module):
    """Hybrid quantum‑classical sequence‑tagger with optional quantum convolution and estimator."""
    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 n_qubits: int = 0,
                 conv_kernel: int | None = None,
                 fraud_params: tuple | None = None) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Quantum LSTM
        if n_qubits > 0:
            self.lstm = QuantumQLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
            # In quantum mode, the regression head outputs a scalar
            self.hidden2tag = nn.Linear(1, tagset_size)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
            self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

        # Optional quantum convolutional filter
        self.conv = QuanvCircuit(kernel_size=conv_kernel) if conv_kernel else None

        # Optional fraud‑detection style feature extractor
        self.fraud = build_fraud_detection_program(*fraud_params) if fraud_params else None

        # Estimator QNN (quantum regression head)
        self.estimator = EstimatorQNNWrapper()

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # sentence shape: (seq_len, batch)
        embeds = self.word_embeddings(sentence)  # (seq_len, batch, embedding_dim)

        # Apply quantum convolution to each time‑step if enabled
        if self.conv:
            # This block is a placeholder; actual implementation would reshape
            # the embeddings into 2D patches and feed them into the quantum filter.
            # For brevity, we skip the detailed conversion.
            pass

        # LSTM forward
        lstm_out, _ = self.lstm(embeds)
        if isinstance(lstm_out, tuple):
            lstm_out, _ = lstm_out

        # Fraud‑detection feature extraction
        if self.fraud:
            lstm_out = self.fraud(lstm_out)

        # Quantum regression head
        reg_out = self.estimator(lstm_out)  # (seq_len*batch,)

        # Reshape for linear layer
        reg_out = reg_out.unsqueeze(-1)  # (seq_len*batch, 1)
        tag_logits = self.hidden2tag(reg_out)  # (seq_len*batch, tagset_size)
        return F.log_softmax(tag_logits, dim=-1)

__all__ = ["HybridQLSTM"]
