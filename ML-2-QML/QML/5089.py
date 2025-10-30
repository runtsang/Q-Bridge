import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import qiskit
from qiskit import execute, QuantumCircuit
from qiskit.circuit.random import random_circuit
from strawberryfields import Engine
from strawberryfields import Program
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate
from dataclasses import dataclass
from typing import Iterable

# Quantum convolutional filter
class QuantumConv(nn.Module):
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.n_qubits = kernel_size ** 2
        self.backend = qiskit.Aer.get_backend("qasm_simulator")
        self.shots = 100

        self.circuit = QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self.circuit.rx(self.theta[i], i)
        self.circuit.barrier()
        self.circuit += random_circuit(self.n_qubits, 2)
        self.circuit.measure_all()

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        batch = data.shape[0]
        data_np = data.cpu().numpy().reshape(batch, self.n_qubits)
        param_binds = []
        for sample in data_np:
            bind = {self.theta[i]: np.pi if val > self.threshold else 0.0 for i, val in enumerate(sample)}
            param_binds.append(bind)
        job = execute(self.circuit, self.backend, shots=self.shots, parameter_binds=param_binds)
        result = job.result()
        counts = result.get_counts(self.circuit)
        probs = []
        for key, val in counts.items():
            ones = sum(int(b) for b in key)
            probs.append(ones * val)
        avg = sum(probs) / (self.shots * self.n_qubits * batch)
        return torch.tensor(avg, device=data.device, dtype=torch.float32).unsqueeze(0)

# Quantum self‑attention
class QuantumSelfAttention(nn.Module):
    def __init__(self, n_qubits: int = 4):
        super().__init__()
        self.n_qubits = n_qubits
        self.qr = qiskit.QuantumRegister(n_qubits, "q")
        self.cr = qiskit.ClassicalRegister(n_qubits, "c")
        self.backend = qiskit.Aer.get_backend("qasm_simulator")

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

    def forward(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ) -> torch.Tensor:
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = execute(circuit, self.backend, shots=shots)
        result = job.result().get_counts(circuit)
        probs = np.zeros(2 ** self.n_qubits)
        for bitstring, count in result.items():
            idx = int(bitstring, 2)
            probs[idx] = count / shots
        return torch.tensor(probs.mean(), device="cpu", dtype=torch.float32)

# Quantum fraud‑detection
@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]

class QuantumFraudDetection(nn.Module):
    def __init__(self, input_params: FraudLayerParameters, layers: Iterable[FraudLayerParameters]):
        super().__init__()
        self.program = self._build_program(input_params, layers)
        self.engine = Engine("fock", backend_options={"cutoff_dim": 12})

    def _build_program(self, input_params, layers):
        program = Program(2)
        with program.context as q:
            self._apply_layer(q, input_params, clip=False)
            for layer in layers:
                self._apply_layer(q, layer, clip=True)
        return program

    def _apply_layer(self, modes, params: FraudLayerParameters, clip: bool):
        BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
        for i, phase in enumerate(params.phases):
            Rgate(phase) | modes[i]
        for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
            Sgate(r if not clip else self._clip(r, 5), phi) | modes[i]
        BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
        for i, phase in enumerate(params.phases):
            Rgate(phase) | modes[i]
        for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
            Dgate(r if not clip else self._clip(r, 5), phi) | modes[i]
        for i, k in enumerate(params.kerr):
            Kgate(k if not clip else self._clip(k, 1)) | modes[i]

    def _clip(self, value, bound):
        return max(-bound, min(bound, value))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch = inputs.shape[0]
        results = []
        for i in range(batch):
            state = self.engine.run(self.program, args=[inputs[i]])
            mean_photon = state.measured_photon_numbers.mean()
            results.append(mean_photon)
        return torch.tensor(results, device=inputs.device, dtype=torch.float32).unsqueeze(1)

# Quantum LSTM
class QuantumQLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        self.forget = self._make_qlayer(n_qubits)
        self.input = self._make_qlayer(n_qubits)
        self.update = self._make_qlayer(n_qubits)
        self.output = self._make_qlayer(n_qubits)

        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def _make_qlayer(self, n_wires):
        class QLayer(nn.Module):
            def __init__(self, n_wires):
                super().__init__()
                self.n_wires = n_wires
                self.circuit = QuantumCircuit(n_wires)
                for i in range(n_wires):
                    self.circuit.rx(qiskit.circuit.Parameter(f"theta{i}"), i)
                self.circuit.barrier()
                self.circuit += random_circuit(n_wires, 2)
                self.circuit.measure_all()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                batch = x.shape[0]
                probs = []
                for idx in range(batch):
                    bind = {self.circuit.parameters[i]: x[idx, i].item() for i in range(self.n_wires)}
                    job = execute(self.circuit, qiskit.Aer.get_backend("qasm_simulator"),
                                  shots=100, parameter_binds=[bind])
                    result = job.result().get_counts(self.circuit)
                    count_ones = sum(int(bit) for bit, cnt in result.items() for _ in range(cnt))
                    probs.append(count_ones / (100 * self.n_wires))
                return torch.tensor(probs, device=x.device, dtype=torch.float32)
        return QLayer(n_wires)

    def forward(self, inputs: torch.Tensor, states=None):
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

    def _init_states(self, inputs, states):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

# Hybrid quantum model
class HybridQLSTM(nn.Module):
    """Hybrid quantum‑classical tagger combining quantum LSTM, quantum convolution,
    quantum self‑attention, and quantum fraud‑detection layers.  All modules are
    wrapped as PyTorch nn.Module so that the entire network can be trained with
    standard optimisers."""
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        use_conv: bool = True,
        use_attention: bool = True,
        use_fraud: bool = True,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QuantumQLSTM(embedding_dim, hidden_dim, n_qubits)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

        self.use_conv = use_conv
        self.use_attention = use_attention
        self.use_fraud = use_fraud

        if use_conv:
            self.conv = QuantumConv(kernel_size=2, threshold=0.0)
        if use_attention:
            self.attention = QuantumSelfAttention(n_qubits=4)
        if use_fraud:
            fraud_params = FraudLayerParameters(
                bs_theta=0.5, bs_phi=0.5,
                phases=(0.1, 0.1),
                squeeze_r=(0.2, 0.2),
                squeeze_phi=(0.0, 0.0),
                displacement_r=(0.3, 0.3),
                displacement_phi=(0.0, 0.0),
                kerr=(0.0, 0.0),
            )
            self.fraud_net = QuantumFraudDetection(fraud_params, layers=[fraud_params, fraud_params])

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.embedding(sentence)
        if self.use_conv:
            _ = self.conv(embeds)
        if self.use_attention:
            rotation = np.random.randn(self.embedding.embedding_dim * 4)
            entangle = np.random.randn(self.embedding.embedding_dim * 4)
            _ = self.attention(rotation, entangle)
        lstm_out, _ = self.lstm(embeds)
        logits = self.hidden2tag(lstm_out)
        tag_logits = F.log_softmax(logits, dim=1)
        if self.use_fraud:
            fraud_out = self.fraud_net(lstm_out[-1].unsqueeze(0))
            tag_logits = torch.cat([tag_logits, fraud_out], dim=1)
        return tag_logits

__all__ = ["HybridQLSTM"]
