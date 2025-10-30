import torch
from torch import nn
import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector, Pauli
import torchquantum as tq
import torchquantum.functional as tqf

class HybridAutoencoder(nn.Module):
    """
    Hybrid auto‑encoder that fuses classical MLPs with a quantum
    fully‑connected layer inspired by FCL, a QCNN‑style feature map,
    and an optional quantum LSTM for sequence data.
    """
    def __init__(self, config, use_quantum=True, use_sequence=False, n_qubits=4):
        super().__init__()
        self.config = config
        self.use_quantum = use_quantum
        self.use_sequence = use_sequence
        self.n_qubits = n_qubits

        # Classical encoder
        self.encoder = self._build_encoder()

        # Quantum latent mapping
        self.qc, self.params = self._build_quantum_circuit()

        # Sequence module
        if self.use_sequence:
            self.lstm = QLSTM(
                input_dim=config.latent_dim,
                hidden_dim=config.latent_dim,
                n_qubits=config.latent_dim,
            )
        else:
            self.lstm = None

        # Classical decoder
        self.decoder = self._build_decoder()

    def _build_encoder(self):
        layers = []
        in_dim = self.config.input_dim
        for hidden in self.config.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.ReLU())
            if self.config.dropout > 0.0:
                layers.append(nn.Dropout(self.config.dropout))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, self.config.latent_dim))
        return nn.Sequential(*layers)

    def _build_decoder(self):
        layers = []
        in_dim = self.config.latent_dim
        for hidden in reversed(self.config.hidden_dims):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.ReLU())
            if self.config.dropout > 0.0:
                layers.append(nn.Dropout(self.config.dropout))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, self.config.input_dim))
        return nn.Sequential(*layers)

    def _build_quantum_circuit(self):
        """
        Build a simple variational circuit that emulates a fully
        connected quantum layer.  It consists of an H gate and a
        parameterized Ry for each qubit.  The expectation value
        of PauliZ on each qubit is returned as the latent vector.
        """
        qc = QuantumCircuit(self.n_qubits)
        params = ParameterVector('theta', self.n_qubits)
        for q in range(self.n_qubits):
            qc.h(q)
            qc.ry(params[q], q)
        return qc, params

    def _quantum_latent(self, encoded):
        """
        Evaluate the quantum circuit for each sample in the batch and
        return expectation values of PauliZ on all qubits.
        """
        batch = encoded.shape[0]
        expectations = []
        backend = Aer.get_backend('statevector_simulator')
        for i in range(batch):
            param_vals = encoded[i].detach().cpu().numpy()
            bound_qc = self.qc.bind_parameters(dict(zip(self.params, param_vals)))
            job = execute(bound_qc, backend)
            state = job.result().get_statevector()
            sv = Statevector(state)
            exp_vals = []
            for q in range(self.n_qubits):
                pauli_str = 'I'*q + 'Z' + 'I'*(self.n_qubits-q-1)
                exp = sv.expectation_value(Pauli(pauli_str))
                exp_vals.append(exp)
            expectations.append(exp_vals)
        return torch.tensor(expectations, dtype=torch.float32, device=encoded.device)

    def forward(self, x):
        """
        Forward pass.  For static data x has shape (batch, input_dim).
        For sequences x has shape (batch, seq_len, input_dim) when
        `use_sequence=True`.
        """
        if self.use_sequence:
            batch, seq_len, _ = x.shape
            flat = x.reshape(batch * seq_len, -1)
            encoded = self.encoder(flat)
            if self.use_quantum:
                latent = self._quantum_latent(encoded)
            else:
                latent = encoded
            latent = latent.reshape(batch, seq_len, -1)
            lstm_out, _ = self.lstm(latent)
            decoded = self.decoder(lstm_out)
            return decoded.reshape(batch, seq_len, -1)
        else:
            encoded = self.encoder(x)
            if self.use_quantum:
                latent = self._quantum_latent(encoded)
            else:
                latent = encoded
            decoded = self.decoder(latent)
            return decoded

# Quantum LSTM cell from reference pair 4 (torchquantum)
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
        self.params = nn.ModuleList(
            [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x):
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

class QLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
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

    def forward(self, inputs, states=None):
        # inputs shape: (batch, seq_len, input_dim)
        inputs = inputs.transpose(0, 1)  # (seq_len, batch, input_dim)
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
        outputs = torch.cat(outputs, dim=0)  # (seq_len, batch, hidden_dim)
        outputs = outputs.transpose(0, 1)   # (batch, seq_len, hidden_dim)
        return outputs, (hx, cx)

    def _init_states(self, inputs, states):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )
