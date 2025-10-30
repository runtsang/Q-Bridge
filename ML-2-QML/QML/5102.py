"""Quantum hybrid classifier combining variational circuits, quantum LSTM, Qiskit self‑attention, and TorchQuantum kernel."""
from __future__ import annotations

from typing import Iterable, Tuple, Sequence

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from torchquantum.functional import func_name_dict as tqf
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.providers.aer import AerSimulator

# --------------------------------------------------------------------
# 1. Variational classifier circuit (Qiskit)
# --------------------------------------------------------------------
def build_classifier_circuit(num_qubits: int, depth: int = 3) -> Tuple[QuantumCircuit, Iterable, Iterable, list[str]]:
    """
    Construct a layered variational ansatz that mirrors the classical feed‑forward interface.

    Returns
    -------
    circuit : QuantumCircuit
        A Qiskit circuit ready for simulation.
    encoding : list[ParameterVector]
        Parameter vectors for feature encoding.
    weights : list[ParameterVector]
        Parameter vectors for variational layers.
    observables : list[str]
        Measurement operators used as logits.
    """
    encoding = tq.ParameterVector("x", num_qubits)
    weights = tq.ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        # entanglement layer
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    observables = [f"I" * i + "Z" + f"I" * (num_qubits - i - 1) for i in range(num_qubits)]
    return circuit, list(encoding), list(weights), observables

# --------------------------------------------------------------------
# 2. Quantum self‑attention (Qiskit implementation)
# --------------------------------------------------------------------
class QuantumSelfAttention:
    """
    Qiskit based self‑attention block that encodes query/key/value rotations
    and measures a simple correlation pattern.
    """

    def __init__(self, n_qubits: int) -> None:
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")
        self.backend = Aer.get_backend("qasm_simulator")

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        circuit = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        circuit.measure(self.qr, self.cr)
        return circuit

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, shots: int = 1024) -> dict:
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = execute(circuit, self.backend, shots=shots)
        return job.result().get_counts(circuit)

# --------------------------------------------------------------------
# 3. Quantum kernel (TorchQuantum)
# --------------------------------------------------------------------
class KernalAnsatz(tq.QuantumModule):
    """
    TorchQuantum circuit that encodes two classical vectors into quantum states
    and applies a symmetric gate sequence.
    """

    def __init__(self, func_list: Sequence[dict]) -> None:
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

class Kernel(tq.QuantumModule):
    """
    Quantum kernel evaluated via a fixed TorchQuantum ansatz.
    """

    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = KernalAnsatz(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    """
    Compute Gram matrix using the quantum kernel.
    """
    kernel = Kernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# --------------------------------------------------------------------
# 4. Quantum LSTM (TorchQuantum implementation)
# --------------------------------------------------------------------
class QLSTM(tq.QuantumModule):
    """
    LSTM cell where each gate is realised by a small quantum circuit.
    """

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
            self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
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

    def forward(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None = None):
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

# --------------------------------------------------------------------
# 5. Quantum hybrid classifier model
# --------------------------------------------------------------------
class QuantumHybridClassifier(tq.QuantumModule):
    """
    Quantum counterpart of :class:`QuantumHybridClassifier` from the classical module.
    It couples a variational classifier ansatz, a quantum LSTM encoder, a Qiskit self‑attention block,
    and a TorchQuantum kernel into a single differentiable module.
    """

    def __init__(
        self,
        num_qubits: int,
        lstm_hidden: int = 64,
        lstm_depth: int = 1,
        use_attention: bool = True,
    ) -> None:
        super().__init__()
        self.num_qubits = num_qubits
        # Variational classifier circuit (kept for reference, not used directly in the forward)
        self.classifier_circuit, self.encoding, self.weights, self.observables = build_classifier_circuit(num_qubits, depth=3)
        self.lstm = QLSTM(input_dim=num_qubits, hidden_dim=lstm_hidden, n_qubits=num_qubits)
        self.use_attention = use_attention
        if use_attention:
            self.attention = QuantumSelfAttention(n_qubits=num_qubits)
        self.kernel = Kernel()
        # Classical linear head to produce logits
        self.classifier = nn.Linear(num_qubits + 1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the quantum hybrid architecture.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, num_qubits).

        Returns
        -------
        torch.Tensor
            Log‑softmax logits of shape (batch, seq_len, 2).
        """
        # Quantum LSTM encoding
        lstm_out, _ = self.lstm(x)
        # Optional quantum attention
        if self.use_attention:
            rotation = np.random.rand(3 * self.num_qubits)
            entangle = np.random.rand(self.num_qubits - 1)
            attn_counts = self.attention.run(rotation, entangle, shots=512)
            # Convert counts to a simple normalized tensor
            attn_tensor = torch.tensor(
                [list(attn_counts.values())[0] / sum(attn_counts.values())],
                dtype=torch.float32,
                device=x.device,
            ).expand_as(lstm_out)
            lstm_out = lstm_out + attn_tensor
        # Quantum kernel augmentation
        batch_size, seq_len, _ = x.shape
        prototypes = torch.randn(batch_size, seq_len, self.num_qubits, device=x.device)
        kernel_features = torch.stack(
            [self.kernel(x[i, j], prototypes[i, j]) for i in range(batch_size) for j in range(seq_len)]
        ).view(batch_size, seq_len, 1)
        # Concatenate with LSTM output
        combined = torch.cat([lstm_out, kernel_features], dim=-1)
        # Classical linear classifier on top of quantum features
        logits = self.classifier(combined)
        return torch.log_softmax(logits, dim=-1)

__all__ = ["QuantumHybridClassifier", "build_classifier_circuit", "QuantumSelfAttention", "Kernel", "kernel_matrix"]
