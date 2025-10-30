import torch
import torch.nn as nn
import torch.quantum as tq
import torch.quantum.functional as tqf
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
import numpy as np


class QuantumQuanvolutionFilter(tq.QuantumModule):
    """Apply a random two‑qubit quantum kernel to 2×2 image patches."""
    def __init__(self, n_wires: int = 4, patch_size: int = 2) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.patch_size = patch_size
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, h, w = x.size(0), 28, 28  # MNIST dimensions
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        patches = []
        for r in range(0, h, self.patch_size):
            for c in range(0, w, self.patch_size):
                data = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, data)
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)


class QuantumSelfAttention:
    """Basic quantum circuit representing a self‑attention style block."""
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")
        self.backend = qiskit.Aer.get_backend("qasm_simulator")

    def _build_circuit(
        self, rotation_params: np.ndarray, entangle_params: np.ndarray
    ) -> QuantumCircuit:
        circuit = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        circuit.measure(self.qr, self.cr)
        return circuit

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ) -> dict:
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = qiskit.execute(circuit, self.backend, shots=shots)
        return job.result().get_counts(circuit)


class QuantumQLSTM(tq.QuantumModule):
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
            qdev = tq.QuantumDevice(n_wires=self.n_wires,
                                    bsz=x.shape[0],
                                    device=x.device)
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            for wire in range(self.n_wires):
                if wire == self.n_wires - 1:
                    tqf.cnot(qdev, wires=[wire, 0])
                else:
                    tqf.cnot(qdev, wires=[wire, wire + 1])
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
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

    def forward(self, inputs: torch.Tensor, states: tuple = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in torch.unbind(inputs, dim=0):
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

    def _init_states(self, inputs: torch.Tensor, states: tuple = None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )


def build_classifier_circuit(num_qubits: int, depth: int):
    """Construct a layered ansatz with explicit encoding and variational parameters."""
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)
    circuit = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)
    index = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[index], qubit)
            index += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)
    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]
    return circuit, list(encoding), list(weights), observables


class QuanvolutionHybrid(tq.QuantumModule):
    """
    Fully quantum hybrid model that chains:
      1. Quanvolutional feature extraction
      2. Quantum self‑attention
      3. Quantum LSTM sequence modelling
      4. Quantum classifier ansatz
    """
    def __init__(
        self,
        patch_dim: int = 4 * 14 * 14,
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        classifier_depth: int = 2,
        num_qubits: int = 4,
    ) -> None:
        super().__init__()
        self.filter = QuantumQuanvolutionFilter()
        self.attention = QuantumSelfAttention(num_qubits)
        self.lstm = QuantumQLSTM(patch_dim, lstm_hidden, num_qubits)
        self.classifier_circuit, self.enc_params, self.wt_params, self.obs = build_classifier_circuit(num_qubits, classifier_depth)
        # Trainable parameters for attention
        self.rotation = nn.Parameter(torch.randn(num_qubits, 3))
        self.entangle = nn.Parameter(torch.randn(num_qubits - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Quantum patch extraction
        features = self.filter(x)

        # 2. Reshape into sequence of patch vectors
        seq = features.view(features.size(0), -1, 4)

        # 3. Quantum self‑attention
        attn_counts = self.attention.run(
            rotation_params=self.rotation.detach().cpu().numpy(),
            entangle_params=self.entangle.detach().cpu().numpy(),
            shots=512,
        )
        # Convert counts to a dense feature tensor
        attn_tensor = torch.tensor(
            np.array(list(attn_counts.values())),
            dtype=x.dtype,
            device=x.device,
        ).unsqueeze(0)

        # 4. Quantum LSTM sequence modelling
        lstm_out, _ = self.lstm(attn_tensor)

        # 5. Quantum classifier ansatz (expectation value)
        self.classifier_circuit.bind_parameters({**dict(zip(self.enc_params, torch.arange(num_qubits))), **dict(zip(self.wt_params, torch.linspace(0, 1, len(self.wt_params))))})
        exp_vals = []
        for qubit in range(self.classifier_circuit.num_qubits):
            exp = self.classifier_circuit.expectation(self.obs[qubit], shots=256)
            exp_vals.append(exp)
        logits = torch.stack(exp_vals, dim=1)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionHybrid"]
