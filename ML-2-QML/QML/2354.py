import numpy as np
import qiskit
from qiskit.circuit.random import random_circuit
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class ConvQLSTM(nn.Module):
    """
    Quantum-enhanced Conv-QLSTM module.
    Uses a quantum convolution filter (QuanvFilter) and a quantum LSTM (QLayer)
    to process input sequences. The filter outputs a probability distribution
    that is fed into the quantum LSTM gates, which are realized by small
    variational circuits. The final tag logits are produced by a classical
    linear layer.
    """
    class QuanvFilter(nn.Module):
        """
        Quantum filter that encodes a 2D kernel into a parameterised RX circuit.
        """
        def __init__(self, kernel_size: int = 2, threshold: float = 127, shots: int = 100):
            super().__init__()
            self.kernel_size = kernel_size
            self.threshold = threshold
            self.n_qubits = kernel_size ** 2
            self.shots = shots
            self.backend = qiskit.Aer.get_backend("qasm_simulator")
            self._circuit = qiskit.QuantumCircuit(self.n_qubits)
            self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
            for i in range(self.n_qubits):
                self._circuit.rx(self.theta[i], i)
            self._circuit.barrier()
            self._circuit += random_circuit(self.n_qubits, 2)
            self._circuit.measure_all()

        def forward(self, data: torch.Tensor) -> torch.Tensor:
            """
            data: Tensor of shape (batch, kernel_size, kernel_size)
            Returns: Tensor of shape (batch, n_qubits) with average |1> probabilities.
            """
            batch = data.shape[0]
            data_flat = data.reshape(batch, -1)
            param_binds = []
            for row in data_flat:
                bind = {}
                for i, val in enumerate(row):
                    bind[self.theta[i]] = np.pi if val > self.threshold else 0.0
                param_binds.append(bind)
            job = qiskit.execute(self._circuit, self.backend,
                                 shots=self.shots,
                                 parameter_binds=param_binds)
            result = job.result()
            counts = result.get_counts(self._circuit)
            probs = []
            for key, val in counts.items():
                ones = sum(int(b) for b in key)
                probs.append((ones / self.n_qubits) * (val / self.shots))
            probs = np.array(probs)
            return torch.tensor(probs, dtype=torch.float32)

    class QLayer(tq.QuantumModule):
        """
        Quantum layer used as a gate in the quantum LSTM.
        """
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
            for wire in range(self.n_wires - 1):
                tqf.cnot(qdev, wires=[wire, wire + 1])
            tqf.cnot(qdev, wires=[self.n_wires - 1, 0])
            return self.measure(qdev)

    def __init__(self, kernel_size: int = 2, threshold: float = 127, shots: int = 100,
                 hidden_dim: int = 128, vocab_size: int = 5000, tagset_size: int = 10):
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.shots = shots
        self.n_qubits = kernel_size ** 2
        self.conv_filter = self.QuanvFilter(kernel_size, threshold, shots)
        # Quantum LSTM gates
        self.forget_gate = self.QLayer(self.n_qubits)
        self.input_gate = self.QLayer(self.n_qubits)
        self.update_gate = self.QLayer(self.n_qubits)
        self.output_gate = self.QLayer(self.n_qubits)
        # Linear layers to map classical input to n_qubits
        self.linear_forget = nn.Linear(1, self.n_qubits)
        self.linear_input = nn.Linear(1, self.n_qubits)
        self.linear_update = nn.Linear(1, self.n_qubits)
        self.linear_output = nn.Linear(1, self.n_qubits)
        # Classical linear layer to produce tag logits
        self.hidden2tag = nn.Linear(self.n_qubits, tagset_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor of shape (batch, seq_len, 1, kernel_size, kernel_size)
        Returns: log-probabilities over tags for each time step.
        """
        batch, seq_len, c, h, w = x.shape
        # Reshape to process each time step independently
        x_reshaped = x.view(batch * seq_len, c, h, w)
        conv_out = self.conv_filter(x_reshaped)  # shape: (batch*seq_len, n_qubits)
        conv_out = conv_out.view(batch, seq_len, -1)  # (batch, seq_len, n_qubits)
        # Initialize hidden and cell states
        hx = torch.zeros(batch, self.n_qubits, device=x.device)
        cx = torch.zeros(batch, self.n_qubits, device=x.device)
        outputs = []
        for t in range(seq_len):
            x_t = conv_out[:, t, :]  # (batch, n_qubits)
            # Gates
            f = torch.sigmoid(self.forget_gate(self.linear_forget(x_t)))
            i = torch.sigmoid(self.input_gate(self.linear_input(x_t)))
            g = torch.tanh(self.update_gate(self.linear_update(x_t)))
            o = torch.sigmoid(self.output_gate(self.linear_output(x_t)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(1))
        lstm_out = torch.cat(outputs, dim=1)  # (batch, seq_len, n_qubits)
        tag_logits = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_logits, dim=-1)

__all__ = ["ConvQLSTM"]
