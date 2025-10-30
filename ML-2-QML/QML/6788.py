import numpy as np
import qiskit
import torch
from torch import nn
import torchquantum as tq
import torchquantum.functional as tqf

class QuantumFCL(nn.Module):
    """Parameterized Qiskit circuit that outputs a scalar expectation."""
    def __init__(self, n_qubits: int, backend=None, shots: int = 100):
        super().__init__()
        self.n_qubits = n_qubits
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.circuit = self._build_circuit()

    def _build_circuit(self) -> qiskit.QuantumCircuit:
        circ = qiskit.QuantumCircuit(self.n_qubits)
        theta = qiskit.circuit.Parameter("theta")
        circ.h(range(self.n_qubits))
        circ.barrier()
        circ.ry(theta, range(self.n_qubits))
        circ.measure_all()
        self.theta = theta
        return circ

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        thetas = x.squeeze(1).cpu().numpy()
        probs = []
        for theta_val in thetas:
            bound = {self.theta: theta_val}
            job = qiskit.execute(self.circuit, self.backend, shots=self.shots,
                                 parameter_binds=[bound])
            result = job.result().get_counts(self.circuit)
            counts = np.array(list(result.values()))
            states = np.array([int(k, 2) for k in result.keys()])
            probs.append(np.sum(states * counts) / self.shots)
        return torch.tensor(probs, device=x.device).unsqueeze(1)

class QuantumQLSTM(nn.Module):
    """Quantum‑enhanced LSTM cell where each gate is a small circuit."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
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
                tgt = 0 if wire == self.n_wires - 1 else wire + 1
                tqf.cnot(qdev, wires=[wire, tgt])
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
        return torch.cat(outputs, dim=0), (hx, cx)

    def _init_states(self, inputs, states):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

class UnifiedFCLQLSTM(nn.Module):
    """
    Fully quantum variant of the hybrid architecture.  Both the
    fully‑connected layer and the LSTM gates are realized with
    parameterized quantum circuits.
    """
    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 n_qubits: int,
                 backend=None,
                 shots: int = 100):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.fcl = QuantumFCL(n_qubits=n_qubits, backend=backend, shots=shots)
        self.lstm = QuantumQLSTM(input_dim=1,
                                 hidden_dim=hidden_dim,
                                 n_qubits=n_qubits)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)          # (seq_len, batch, embedding_dim)
        fcl_out = self.fcl(embeds)                       # (seq_len, batch, 1)
        lstm_out, _ = self.lstm(fcl_out)
        logits = self.hidden2tag(lstm_out)
        return torch.nn.functional.log_softmax(logits, dim=1)

__all__ = ["UnifiedFCLQLSTM"]
