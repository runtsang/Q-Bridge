import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
import qiskit
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

# Quantum LSTM cell (from reference 3)
class QLSTM(nn.Module):
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [i], "func": "rx", "wires": [i]}
                    for i in range(n_wires)
                ]
            )
            self.params = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            qdev = tq.QuantumDevice(
                n_wires=self.n_wires, bsz=x.shape[0], device=x.device
            )
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

    def forward(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        batch_size = inputs.size(0)
        seq_len = inputs.size(1)
        for t in range(seq_len):
            x = inputs[:, t, :]
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(self.linear_forget(combined)))
            i = torch.sigmoid(self.input(self.linear_input(combined)))
            g = torch.tanh(self.update(self.linear_update(combined)))
            o = torch.sigmoid(self.output(self.linear_output(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(1))
        outputs = torch.cat(outputs, dim=1)
        return outputs, (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(0)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

# Helper to build a quantum convolution circuit (from reference 1)
def build_quantum_conv(kernel_size: int, threshold: float):
    n_qubits = kernel_size ** 2
    backend = qiskit.Aer.get_backend("qasm_simulator")
    circuit = QuantumCircuit(n_qubits)
    theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(n_qubits)]
    for i in range(n_qubits):
        circuit.rx(theta[i], i)
    circuit.barrier()
    circuit += qiskit.circuit.random.random_circuit(n_qubits, 2)
    circuit.measure_all()

    class ConvCircuit:
        def __init__(self):
            self.circuit = circuit
            self.backend = backend
            self.shots = 100
            self.threshold = threshold
            self.n_qubits = n_qubits

        def run(self, data: np.ndarray) -> float:
            data = np.reshape(data, (1, self.n_qubits))
            param_binds = []
            for dat in data:
                bind = {}
                for i, val in enumerate(dat):
                    bind[theta[i]] = np.pi if val > self.threshold else 0
                param_binds.append(bind)
            job = qiskit.execute(
                self.circuit,
                self.backend,
                shots=self.shots,
                parameter_binds=param_binds,
            )
            result = job.result().get_counts(self.circuit)
            counts = 0
            for key, val in result.items():
                ones = sum(int(bit) for bit in key)
                counts += ones * val
            return counts / (self.shots * self.n_qubits)

    return ConvCircuit()

# Helper to build a quantum fully‑connected layer (from reference 2)
def build_quantum_fc(num_qubits: int):
    backend = qiskit.Aer.get_backend("qasm_simulator")
    circuit = QuantumCircuit(num_qubits)
    theta = qiskit.circuit.Parameter("theta")
    circuit.h(range(num_qubits))
    circuit.barrier()
    circuit.ry(theta, range(num_qubits))
    circuit.measure_all()

    class FCCircuit:
        def __init__(self):
            self.circuit = circuit
            self.backend = backend
            self.shots = 100

        def run(self, thetas: list[float]) -> np.ndarray:
            job = qiskit.execute(
                self.circuit,
                self.backend,
                shots=self.shots,
                parameter_binds=[{theta: t} for t in thetas],
            )
            result = job.result().get_counts(self.circuit)
            counts = np.array(list(result.values()))
            states = np.array(list(result.keys())).astype(float)
            probabilities = counts / self.shots
            expectation = np.sum(states * probabilities)
            return np.array([expectation])

    return FCCircuit()

# Helper to build a quantum classifier (from reference 4)
def build_quantum_classifier(num_qubits: int, depth: int):
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)
    circuit = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)
    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)
    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]

    class ClassifierCircuit:
        def __init__(self):
            self.circuit = circuit
            self.backend = qiskit.Aer.get_backend("qasm_simulator")
            self.shots = 100
            self.weights = weights
            self.observables = observables

        def run(self, data: np.ndarray) -> np.ndarray:
            # Resize data to match weight count
            data = np.resize(data, len(self.weights))
            param_binds = [{self.weights[i]: val for i, val in enumerate(data)}]
            job = qiskit.execute(
                self.circuit,
                self.backend,
                shots=self.shots,
                parameter_binds=param_binds,
            )
            result = job.result().get_counts(self.circuit)
            # compute expectation for each observable
            exp_vals = []
            for obs in self.observables:
                exp = 0.0
                for key, val in result.items():
                    eigen = 1
                    for qubit, bit in enumerate(reversed(key)):
                        if obs.to_label()[qubit] == "Z" and bit == "1":
                            eigen *= -1
                    exp += eigen * val
                exp_vals.append(exp / self.shots)
            return np.array(exp_vals)

    return ClassifierCircuit()

# Main hybrid module
class ConvGen126(nn.Module):
    """
    Hybrid quantum‑classical module that mirrors the original Conv.py
    but includes additional fully‑connected, LSTM, and classifier
    components. The quantum implementation builds variational circuits
    for each sub‑module and measures observables to produce classical
    outputs.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.0,
        num_qubits: int = 4,
        depth: int = 2,
        num_features: int = 10,
        hidden_dim: int = 20,
        vocab_size: int = 100,
        tagset_size: int = 5,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.num_qubits = num_qubits
        self.depth = depth

        # Quantum sub‑modules
        self.conv = build_quantum_conv(kernel_size, threshold)
        self.fc = build_quantum_fc(num_qubits)
        self.lstm = QLSTM(1, hidden_dim, num_qubits)
        self.classifier = build_quantum_classifier(num_qubits, depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for quantum mode.

        Args:
            x: Tensor of shape (batch, 1, H, W) where H=W=kernel_size.

        Returns:
            logits: Tensor of shape (batch, 2)
        """
        batch_size = x.shape[0]
        # Convolution
        conv_out = []
        for i in range(batch_size):
            conv_out.append(self.conv.run(x[i, 0].cpu().numpy()))
        conv_out = np.array(conv_out)

        # Fully‑connected
        fc_out = []
        for val in conv_out:
            fc_out.append(self.fc.run([val]))
        fc_out = np.concatenate(fc_out, axis=0)

        # LSTM expects tensor shape (batch, seq_len, features)
        lstm_in = torch.tensor(fc_out, dtype=torch.float32).unsqueeze(1)
        lstm_out, _ = self.lstm(lstm_in)
        # Classifier
        logits = []
        for out in lstm_out.detach().cpu().numpy():
            logits.append(self.classifier.run(out))
        logits = np.concatenate(logits, axis=0)
        return torch.tensor(logits, dtype=torch.float32)
