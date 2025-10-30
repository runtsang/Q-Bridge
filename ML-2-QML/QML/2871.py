import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
import qiskit

class SelfAttentionQLSTM(nn.Module):
    """
    Quantum‑enhanced self‑attention + LSTM module.
    The self‑attention block is a parameterized Qiskit circuit that outputs a
    probability distribution over basis states.  The LSTM gates are realised
    by small variational circuits implemented with torchquantum.
    """
    def __init__(self, embed_dim: int, hidden_dim: int, n_qubits: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Quantum self‑attention parameters
        self.rotation_params = nn.Parameter(torch.randn(n_qubits * 3))
        self.entangle_params  = nn.Parameter(torch.randn(n_qubits - 1))

        # Quantum‑enhanced LSTM
        self.qlstm = QLSTM(embed_dim, hidden_dim, n_qubits)

    def _build_q_attention_circuit(self, rotation_params, entangle_params):
        qr = qiskit.QuantumRegister(self.n_qubits, "q")
        cr = qiskit.ClassicalRegister(self.n_qubits, "c")
        circuit = qiskit.QuantumCircuit(qr, cr)

        # Rotation layer
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)

        # Entanglement layer
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)

        circuit.measure_all()
        return circuit

    def forward(self, inputs: torch.Tensor, backend=None, shots: int = 1024) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs : torch.Tensor
            Input sequence of shape (seq_len, batch, embed_dim).  The embed_dim
            dimension is ignored for the quantum attention; the circuit outputs
            a probability vector of size 2**n_qubits which is then used as the
            attention representation.
        backend : qiskit.providers.basebackend.BaseBackend, optional
            Qiskit simulator or real backend.  Defaults to Aer qasm_simulator.
        shots : int, optional
            Number of shots for the quantum simulation.

        Returns
        -------
        torch.Tensor
            Output sequence of shape (seq_len, batch, hidden_dim).
        """
        seq_len, batch, _ = inputs.shape
        attn_vectors = []

        if backend is None:
            backend = qiskit.Aer.get_backend("qasm_simulator")

        for _ in range(seq_len):
            circ = self._build_q_attention_circuit(self.rotation_params, self.entangle_params)
            job = qiskit.execute(circ, backend=backend, shots=shots)
            counts = job.result().get_counts(circ)

            # Convert counts to probability vector
            probs = np.array([counts.get(bin(i)[2:].zfill(self.n_qubits), 0) for i in range(2**self.n_qubits)])
            probs = probs / shots
            probs = torch.tensor(probs, dtype=torch.float32, device=inputs.device)

            # Expand to batch dimension
            attn_vectors.append(probs.unsqueeze(0).repeat(batch, 1))

        attn_out = torch.stack(attn_vectors, dim=0)  # (seq_len, batch, 2**n_qubits)

        # Quantum LSTM encoding
        lstm_out, _ = self.qlstm(attn_out)
        return lstm_out

class QLSTM(nn.Module):
    """
    LSTM cell where each gate is a variational quantum circuit.
    Mirrors the structure from the seed QLSTM implementation.
    """
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [0], "func": "rx", "wires": [0]},
                 {"input_idx": [1], "func": "rx", "wires": [1]},
                 {"input_idx": [2], "func": "rx", "wires": [2]},
                 {"input_idx": [3], "func": "rx", "wires": [3]}]
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

    def forward(self, inputs: torch.Tensor, states: tuple[torch.Tensor, torch.Tensor] = None):
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

    def _init_states(self, inputs: torch.Tensor, states: tuple[torch.Tensor, torch.Tensor] | None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))
