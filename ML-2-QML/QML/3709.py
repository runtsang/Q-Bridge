import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import qiskit
from qiskit.circuit.random import random_circuit
from qiskit import Aer, execute
import torchquantum as tq
import torchquantum.functional as tqf

class ConvFilterQ(tq.QuantumModule):
    """
    Quantum convolutional filter that operates on a single patch.
    Each patch is encoded into `n_qubits` qubits, a parameterized circuit is run,
    and the average probability of measuring |1> is returned.
    """
    def __init__(self, kernel_size, backend, shots, threshold):
        super().__init__()
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.backend = backend
        self.shots = shots
        self.threshold = threshold

        # Build a random circuit that will be parameterized by data values
        self._circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()

    def forward(self, data):
        """
        Parameters
        ----------
        data : torch.Tensor
            Tensor of shape (batch, kernel_size, kernel_size) with values in [0, 255].
        Returns
        -------
        torch.Tensor
            Tensor of shape (batch,) containing the average |1> probability per batch.
        """
        batch = data.shape[0]
        # Flatten each patch
        flat = data.view(batch, -1).cpu().numpy()

        # Bind parameters based on threshold
        param_binds = []
        for vec in flat:
            bind = {self.theta[i]: np.pi if val > self.threshold else 0.0 for i, val in enumerate(vec)}
            param_binds.append(bind)

        job = execute(self._circuit,
                      self.backend,
                      shots=self.shots,
                      parameter_binds=param_binds)
        result = job.result().get_counts(self._circuit)

        probs = []
        for key, count in result.items():
            ones = key.count('1')
            probs.append(ones * count / (self.shots * self.n_qubits))
        return torch.tensor(probs, dtype=torch.float32, device=data.device)

class QLayer(tq.QuantumModule):
    """
    Small quantum gate block used inside the quantum LSTM cell.
    """
    def __init__(self, n_wires):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
        )
        self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x):
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
        self.encoder(qdev, x)
        for gate in self.params:
            gate(qdev)
        for i in range(self.n_wires - 1):
            tqf.cnot(qdev, wires=[i, i + 1])
        tqf.cnot(qdev, wires=[self.n_wires - 1, 0])
        return self.measure(qdev)

class QLSTMCell(tq.QuantumModule):
    """
    Quantum‑enhanced LSTM cell that uses QLayer for each gate.
    """
    def __init__(self, input_dim, hidden_dim, n_qubits):
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

    def forward(self, x, hx, cx):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input at the current time step, shape (batch, input_dim).
        hx : torch.Tensor
            Hidden state from previous step, shape (batch, hidden_dim).
        cx : torch.Tensor
            Cell state from previous step, shape (batch, hidden_dim).
        Returns
        -------
        hx, cx : torch.Tensor
            Updated hidden and cell states.
        """
        combined = torch.cat([x, hx], dim=1)
        f = torch.sigmoid(self.forget(self.linear_forget(combined)))
        i = torch.sigmoid(self.input(self.linear_input(combined)))
        g = torch.tanh(self.update(self.linear_update(combined)))
        o = torch.sigmoid(self.output(self.linear_output(combined)))
        cx = f * cx + i * g
        hx = o * torch.tanh(cx)
        return hx, cx

class ConvQLSTM(tq.QuantumModule):
    """
    Hybrid module that combines a quantum convolutional filter with a quantum LSTM cell.
    If ``n_qubits`` is set to 0, the module falls back to classical Conv2d and LSTM.
    """
    def __init__(self, kernel_size=2, threshold=127, n_qubits=0,
                 hidden_dim=128, vocab_size=30522):
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.n_qubits = n_qubits

        if self.n_qubits > 0:
            # Quantum convolution filter
            self.filter = ConvFilterQ(kernel_size=kernel_size,
                                      backend=Aer.get_backend('qasm_simulator'),
                                      shots=100,
                                      threshold=threshold)
            self.lstm_cell = QLSTMCell(input_dim=kernel_size*kernel_size,
                                       hidden_dim=hidden_dim,
                                       n_qubits=n_qubits)
        else:
            # Classical fallback
            self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)
            self.lstm = nn.LSTM(input_size=kernel_size*kernel_size,
                                hidden_size=hidden_dim,
                                batch_first=True)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 1, H, W).
        Returns
        -------
        torch.Tensor
            LSTM outputs: (batch, seq_len, hidden_dim).
        """
        batch, _, H, W = x.shape

        # Extract non‑overlapping patches
        patches = x.unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1)
        patches = patches.contiguous().view(batch, -1, self.kernel_size, self.kernel_size)

        if self.n_qubits > 0:
            # Apply quantum filter to each patch
            patch_vals = []
            for i in range(patches.size(1)):
                patch_vals.append(self.filter(patches[:, i]))
            patch_vals = torch.stack(patch_vals, dim=1)  # (batch, seq_len)
            # Expand to match LSTM input dimension
            patch_vals = patch_vals.unsqueeze(-1).repeat(1, 1, self.kernel_size*self.kernel_size)
            seq = patch_vals
            # Iterate through sequence with quantum LSTM cell
            hx = torch.zeros(batch, self.lstm_cell.hidden_dim, device=x.device)
            cx = torch.zeros(batch, self.lstm_cell.hidden_dim, device=x.device)
            outputs = []
            for t in range(seq.size(1)):
                hx, cx = self.lstm_cell(seq[:, t], hx, cx)
                outputs.append(hx.unsqueeze(1))
            outputs = torch.cat(outputs, dim=1)
            return outputs
        else:
            # Classical path
            conv_out = self.conv(patches.unsqueeze(1)).squeeze(1)  # shape (batch, seq_len, 1)
            conv_out = conv_out.view(batch, conv_out.size(1), -1)
            out, _ = self.lstm(conv_out)
            return out

__all__ = ["ConvQLSTM"]
