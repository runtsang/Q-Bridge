import torch
import torch.nn as nn
import numpy as np
import qiskit
import torchquantum as tq
import torchquantum.functional as tqf

class QuantumFCL(nn.Module):
    """Parameterised quantum circuit emulating a fully‑connected layer."""
    def __init__(self, n_qubits: int = 1, shots: int = 512) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = qiskit.Aer.get_backend("qasm_simulator")
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        self.theta = qiskit.circuit.Parameter("theta")
        self.circuit.h(range(n_qubits))
        self.circuit.barrier()
        self.circuit.ry(self.theta, range(n_qubits))
        self.circuit.measure_all()

    def forward(self, thetas: torch.Tensor) -> torch.Tensor:
        job = qiskit.execute(self.circuit, self.backend,
                             shots=self.shots,
                             parameter_binds=[{self.theta: t} for t in thetas])
        result = job.result()
        counts = result.get_counts(self.circuit)
        probs = np.array(list(counts.values())) / self.shots
        states = np.array(list(counts.keys()), dtype=float)
        expectation = np.sum(states * probs)
        return torch.tensor([expectation], dtype=torch.float32)

class QuantumSelfAttention(nn.Module):
    """Quantum self‑attention via a parameterised circuit."""
    def __init__(self, n_qubits: int = 4, shots: int = 512) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = qiskit.Aer.get_backend("qasm_simulator")
        self.qr = qiskit.QuantumRegister(n_qubits, "q")
        self.cr = qiskit.ClassicalRegister(n_qubits, "c")

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> qiskit.QuantumCircuit:
        circ = qiskit.QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            circ.rx(rotation_params[3 * i], i)
            circ.ry(rotation_params[3 * i + 1], i)
            circ.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circ.crx(entangle_params[i], i, i + 1)
        circ.measure(self.qr, self.cr)
        return circ

    def forward(self,
                rotation_params: np.ndarray,
                entangle_params: np.ndarray,
                inputs: torch.Tensor) -> torch.Tensor:
        batch_size = inputs.shape[0]
        outputs = []
        for idx in range(batch_size):
            circ = self._build_circuit(rotation_params, entangle_params)
            for i in range(self.n_qubits):
                circ.ry(inputs[idx, i].item(), i)
            job = qiskit.execute(circ, self.backend, shots=self.shots)
            counts = job.result().get_counts(circ)
            probs = np.array(list(counts.values())) / self.shots
            states = np.array(list(counts.keys()), dtype=float)
            expectation = np.sum(states * probs)
            outputs.append(expectation)
        return torch.tensor(outputs, dtype=torch.float32).unsqueeze(1)

class QuantumQLSTM(nn.Module):
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
        self.hidden_dim = hidden_dim
        self.forget = self.QLayer(n_qubits)
        self.input = self.QLayer(n_qubits)
        self.update = self.QLayer(n_qubits)
        self.output = self.QLayer(n_qubits)
        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(self,
                inputs: torch.Tensor,
                states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
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
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def _init_states(self,
                     inputs: torch.Tensor,
                     states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))

class QuantumQCNN(nn.Module):
    """Quantum‑inspired convolutional network using torchquantum."""
    def __init__(self, n_qubits: int = 8, hidden_dim: int = 4) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.hidden_dim = hidden_dim
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.linear = nn.Linear(n_qubits, hidden_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size = inputs.shape[0]
        device = inputs.device
        qdev = tq.QuantumDevice(n_wires=self.n_qubits, bsz=batch_size, device=device)
        for i in range(self.n_qubits):
            tq.RX(inputs[:, i], wires=i)(qdev)
        for _ in range(2):
            for i in range(self.n_qubits - 1):
                tq.CNOT(wires=[i, i + 1])(qdev)
            for i in range(self.n_qubits):
                tq.RZ(torch.rand(1).to(device), wires=i)(qdev)
        out = self.measure(qdev)
        out = self.linear(out)
        return torch.sigmoid(out)

class HybridLayer(nn.Module):
    """Unified interface that chains Quantum FCL → Quantum LSTM → Quantum Self‑Attention → Quantum QCNN."""
    def __init__(self,
                 n_features: int = 1,
                 input_dim: int = 4,
                 hidden_dim: int = 8,
                 n_qubits: int = 4,
                 embed_dim: int = 4) -> None:
        super().__init__()
        self.fcl = QuantumFCL(n_qubits=n_qubits)
        self.lstm = QuantumQLSTM(input_dim, hidden_dim, n_qubits)
        self.attention = QuantumSelfAttention(n_qubits=embed_dim)
        self.cnn = QuantumQCNN(n_qubits=embed_dim)

    def forward(self,
                thetas: torch.Tensor,
                lstm_input: torch.Tensor,
                lstm_states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                rotation_params: np.ndarray = None,
                entangle_params: np.ndarray = None,
                attention_input: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor,
                                                               torch.Tensor, torch.Tensor,
                                                               Tuple[torch.Tensor, torch.Tensor]]:
        fcl_out = self.fcl(thetas)
        lstm_out, lstm_states = self.lstm(lstm_input, lstm_states)
        attention_out = self.attention(rotation_params, entangle_params, attention_input)
        cnn_out = self.cnn(attention_out)
        return fcl_out, lstm_out, attention_out, cnn_out, lstm_states

__all__ = ["HybridLayer"]
