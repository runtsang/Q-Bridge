import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchquantum as tq
import qiskit
from qiskit import Aer

class QConvFilter(tq.QuantumModule):
    """Quantum analogue of ConvFilter.  Encodes a 2‑D patch into qubits,
    applies a shallow variational layer and measures Pauli‑Z."""
    def __init__(self, n_qubits: int, threshold: float = 0.0) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.threshold = threshold
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
        )
        self.var_layer = tq.RandomLayer(n_ops=10, wires=list(range(n_qubits)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
        self.encoder(qdev, qdev.bsz * [0])  # placeholder for actual data
        self.var_layer(qdev)
        return self.measure(qdev)

class QuantumSelfAttention(tq.QuantumModule):
    """Quantum self‑attention block built with Qiskit."""
    def __init__(self, n_qubits: int) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.qr = qiskit.QuantumRegister(n_qubits, "q")
        self.cr = qiskit.ClassicalRegister(n_qubits, "c")

    def _build_circuit(self,
                       rotation_params: np.ndarray,
                       entangle_params: np.ndarray) -> qiskit.QuantumCircuit:
        circuit = qiskit.QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        circuit.measure(self.qr, self.cr)
        return circuit

    def forward(self,
                backend,
                rotation_params: np.ndarray,
                entangle_params: np.ndarray,
                shots: int = 1024) -> torch.Tensor:
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = qiskit.execute(circuit, backend, shots=shots)
        result = job.result().get_counts(circuit)
        vec = torch.zeros(self.n_qubits, device="cpu")
        for bitstring, count in result.items():
            bits = np.array([int(b) for b in bitstring[::-1]])
            vec += torch.tensor(bits, dtype=torch.float32) * count
        vec = vec / (shots)
        return vec

class QLSTMGen507(tq.QuantumModule):
    """Quantum‑enhanced LSTM layer that mirrors the classical QLSTMGen507."""
    class QGate(tq.QuantumModule):
        def __init__(self, n_qubits: int) -> None:
            super().__init__()
            self.n_qubits = n_qubits
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
            )
            self.cnot_chain = tq.CNotChain(n_qubits)
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(qdev, qdev.bsz * [0])  # placeholder
            self.cnot_chain(qdev)
            return self.measure(qdev)

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 n_qubits: int,
                 conv_kernel: int = 2,
                 attention_dim: int = 4) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.conv = QConvFilter(n_qubits, threshold=0.5)
        self.forget = self.QGate(n_qubits)
        self.input = self.QGate(n_qubits)
        self.update = self.QGate(n_qubits)
        self.output = self.QGate(n_qubits)
        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.attention = QuantumSelfAttention(n_qubits)

    def forward(self,
                inputs: torch.Tensor,
                states: tuple[torch.Tensor, torch.Tensor] | None = None) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
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
            attn_vec = self.attention.forward(
                backend=Aer.get_backend("qasm_simulator"),
                rotation_params=np.random.rand(3 * self.n_qubits),
                entangle_params=np.random.rand(self.n_qubits - 1),
                shots=256,
            )
            hx = hx + attn_vec.to(hx.device)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def _init_states(self,
                     inputs: torch.Tensor,
                     states: tuple[torch.Tensor, torch.Tensor] | None) -> tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))

class LSTMTaggerRegressor(tq.QuantumModule):
    """Quantum‑enhanced sequence model that uses QLSTMGen507 as its core."""
    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 n_qubits: int = 0,
                 task: str = "tagging") -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embedding_dim)
        self.core = QLSTMGen507(embedding_dim, hidden_dim,
                                n_qubits=n_qubits)
        self.task = task
        if task == "tagging":
            self.head = nn.Linear(hidden_dim, tagset_size)
        elif task == "regression":
            self.head = nn.Linear(hidden_dim, 1)
        else:
            raise ValueError(f"Unsupported task: {task}")

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.emb(sentence)
        hidden = self.core(embeds)
        logits = self.head(hidden)
        if self.task == "tagging":
            return F.log_softmax(logits, dim=-1)
        else:
            return logits.squeeze(-1)

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Quantum state generator that creates a superposition of |0…0> and |1…1>."""
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
    return states, labels

class RegressionDataset(tq.QuantumModule):
    """Quantum dataset wrapper that returns complex state vectors and labels."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

__all__ = ["QLSTMGen507"]
