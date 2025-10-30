"""Quantum‑enhanced LSTM model with QCNN feature extraction and versatile heads.

The implementation follows the same public API as the classical module but replaces
the linear gates with variational quantum circuits.  The QCNN feature extractor
uses a Qiskit EstimatorQNN, while the heads are either a quantum regression
circuit or a hybrid quantum‑classical layer.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
import qiskit
from qiskit import assemble, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator as EstimatorQiskit
import numpy as np
from qiskit import Aer

# --------------------------------------------------------------------------- #
#  QCNN feature extractor (quantum)
# --------------------------------------------------------------------------- #
class QCNNFeatureExtractorQuantum(nn.Module):
    """Quantum version of the QCNN feature extractor using a small variational circuit."""
    def __init__(self, input_dim: int = 8, n_qubits: int = 8) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.n_qubits = n_qubits

        # Feature map: Ry rotations per input feature
        self.feature_map = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(n_qubits)]
        )

        # Variational block: random layer + RX/RY gates
        self.variational = tq.RandomLayer(n_ops=20, wires=list(range(n_qubits)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)

        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_qubits, bsz=bsz, device=x.device)
        self.feature_map(qdev, x)
        self.variational(qdev)
        for wire in range(self.n_qubits):
            self.rx(qdev, wires=wire)
            self.ry(qdev, wires=wire)
        return self.measure(qdev)


# --------------------------------------------------------------------------- #
#  Quantum LSTM cell
# --------------------------------------------------------------------------- #
class QuantumLSTMCell(tq.QuantumModule):
    """LSTM cell where each gate is a small quantum circuit."""
    class QGate(tq.QuantumModule):
        def __init__(self, n_wires: int):
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
            # Simple entangling chain
            for wire in range(self.n_wires - 1):
                tqf.cnot(qdev, wires=[wire, wire + 1])
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Gates
        self.forget = self.QGate(n_qubits)
        self.input = self.QGate(n_qubits)
        self.update = self.QGate(n_qubits)
        self.output = self.QGate(n_qubits)

        # Linear projections to gate space
        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(self, x: torch.Tensor, hx: torch.Tensor, cx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([x, hx], dim=1)
        f = torch.sigmoid(self.forget(self.linear_forget(combined)))
        i = torch.sigmoid(self.input(self.linear_input(combined)))
        g = torch.tanh(self.update(self.linear_update(combined)))
        o = torch.sigmoid(self.output(self.linear_output(combined)))
        cx = f * cx + i * g
        hx = o * torch.tanh(cx)
        return hx, cx


# --------------------------------------------------------------------------- #
#  Quantum regression head
# --------------------------------------------------------------------------- #
class QuantumRegressionHead(tq.QuantumModule):
    """Variational quantum circuit that maps the hidden state to a scalar."""
    def __init__(self, n_qubits: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(n_qubits)]
        )
        self.var_layer = tq.RandomLayer(n_ops=30, wires=list(range(n_qubits)))
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(n_qubits, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_qubits, bsz=bsz, device=x.device)
        self.encoder(qdev, x)
        self.var_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)


# --------------------------------------------------------------------------- #
#  Hybrid quantum‑classical head (binary classification)
# --------------------------------------------------------------------------- #
class QuantumCircuit:
    """Wrapper around a parametrised two‑qubit circuit executed on Aer."""
    def __init__(self, n_qubits: int, backend, shots: int) -> None:
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        all_qubits = list(range(n_qubits))
        self.theta = qiskit.circuit.Parameter("theta")

        self._circuit.h(all_qubits)
        self._circuit.barrier()
        self._circuit.ry(self.theta, all_qubits)
        self._circuit.measure_all()

        self.backend = backend
        self.shots = shots

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self._circuit, self.backend)
        qobj = assemble(compiled, shots=self.shots,
                        parameter_binds=[{self.theta: theta} for theta in thetas])
        job = self.backend.run(qobj)
        result = job.result().get_counts()

        def expectation(count_dict):
            counts = np.array(list(count_dict.values()))
            states = np.array(list(count_dict.keys())).astype(float)
            probabilities = counts / self.shots
            return np.sum(states * probabilities)

        if isinstance(result, list):
            return np.array([expectation(item) for item in result])
        return np.array([expectation(result)])


class HybridQuantumFunction(torch.autograd.Function):
    """Differentiable interface between PyTorch and the quantum circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.quantum_circuit = circuit
        expectation_z = ctx.quantum_circuit.run(inputs.tolist())
        result = torch.tensor(expectation_z, dtype=torch.float32, device=inputs.device)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        input_values = np.array(inputs.tolist())
        shift = np.ones_like(input_values) * ctx.shift
        gradients = []
        for idx, value in enumerate(input_values):
            expectation_right = ctx.quantum_circuit.run([value + shift[idx]])
            expectation_left = ctx.quantum_circuit.run([value - shift[idx]])
            gradients.append(expectation_right - expectation_left)
        gradients = torch.tensor(gradients, dtype=torch.float32, device=inputs.device)
        return gradients * grad_output, None, None


class HybridQuantum(nn.Module):
    """Hybrid head that forwards activations through a quantum circuit."""
    def __init__(self, in_features: int, backend, shots: int, shift: float) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.quantum_circuit = QuantumCircuit(1, backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Map hidden state to a single angle
        angle = self.linear(inputs).squeeze(-1)
        return HybridQuantumFunction.apply(angle, self.quantum_circuit, self.shift)


# --------------------------------------------------------------------------- #
#  Unified LSTM model (quantum)
# --------------------------------------------------------------------------- #
class QLSTMGen224(nn.Module):
    """Quantum‑enhanced LSTM model with QCNN feature extractor and heads.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input features.
    hidden_dim : int
        Hidden size of the LSTM.
    n_qubits : int
        Number of qubits used in the quantum gates.
    task : str
        One of ``'tagger'``, ``'regression'`` or ``'binary'``.
    feature_dim : int
        Dimensionality of the raw input that is fed into the QCNN feature extractor.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int = 0,
        task: str = "tagger",
        feature_dim: int = 8,
    ) -> None:
        super().__init__()
        self.feature_extractor = QCNNFeatureExtractorQuantum(feature_dim, n_qubits)
        self.lstm_cell = QuantumLSTMCell(input_dim, hidden_dim, n_qubits)
        self.hidden_dim = hidden_dim

        if task == "tagger":
            self.head = nn.Linear(hidden_dim, 1)  # placeholder; overridden in wrapper
        elif task == "regression":
            self.head = QuantumRegressionHead(n_qubits)
        elif task == "binary":
            backend = Aer.get_backend("aer_simulator")
            self.head = HybridQuantum(hidden_dim, backend, shots=100, shift=np.pi / 2)
        else:
            raise ValueError(f"Unsupported task {task}")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        seq_len, batch, _ = inputs.shape
        hx = torch.zeros(batch, self.hidden_dim, device=inputs.device)
        cx = torch.zeros(batch, self.hidden_dim, device=inputs.device)

        outputs = []
        for t in range(seq_len):
            x = inputs[t]
            x = self.feature_extractor(x)
            hx, cx = self.lstm_cell(x, hx, cx)
            outputs.append(hx.unsqueeze(0))
        lstm_out = torch.cat(outputs, dim=0)  # (seq_len, batch, hidden_dim)

        # Flatten for head
        lstm_out_flat = lstm_out.view(-1, self.hidden_dim)
        out = self.head(lstm_out_flat)
        return out.view(seq_len, batch, -1)


# --------------------------------------------------------------------------- #
#  Convenience wrappers
# --------------------------------------------------------------------------- #
class LSTMTaggerGen224(QLSTMGen224):
    """Tagger that returns log‑softmax over a tagset."""
    def __init__(self, input_dim, hidden_dim, n_qubits=0, tagset_size=10, **kwargs):
        super().__init__(input_dim, hidden_dim, n_qubits, task="tagger", **kwargs)
        self.head = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        lstm_out = super().forward(sentence)
        return F.log_softmax(lstm_out, dim=-1)


class RegressionGen224(QLSTMGen224):
    """Regression model that outputs a scalar."""
    def __init__(self, input_dim, hidden_dim, n_qubits=0, **kwargs):
        super().__init__(input_dim, hidden_dim, n_qubits, task="regression", **kwargs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        lstm_out = super().forward(inputs)
        return lstm_out.squeeze(-1)


class BinaryClassifierGen224(QLSTMGen224):
    """Binary classifier that returns a probability pair."""
    def __init__(self, input_dim, hidden_dim, n_qubits=0, **kwargs):
        super().__init__(input_dim, hidden_dim, n_qubits, task="binary", **kwargs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        probs = super().forward(inputs)
        return torch.cat((probs, 1 - probs), dim=-1)


__all__ = [
    "QLSTMGen224",
    "LSTMTaggerGen224",
    "RegressionGen224",
    "BinaryClassifierGen224",
    "HybridQuantum",
    "HybridQuantumFunction",
    "QuantumRegressionHead",
]
