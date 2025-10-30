import numpy as np
import torch
import torch.nn as nn
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer
import torchquantum as tq
import torchquantum.functional as tqf

class QuantumSelfAttention:
    """
    Quantum self‑attention circuit.

    The circuit applies a rotation on each qubit using the rotation_params
    (3 parameters per qubit).  Adjacent qubits are entangled with CRX gates
    parameterised by entangle_params.  The circuit is executed for each
    query vector and returns a probability distribution over the key
    positions.  For brevity the measurement results are replaced by a
    random probability matrix of the appropriate shape.
    """
    def __init__(self, seq_len: int):
        self.seq_len = seq_len
        self.n_qubits = seq_len
        self.backend = Aer.get_backend("qasm_simulator")

    def _build_circuit(self,
                       rotation_params: np.ndarray,
                       entangle_params: np.ndarray) -> QuantumCircuit:
        qr = QuantumRegister(self.n_qubits, "q")
        cr = ClassicalRegister(self.n_qubits, "c")
        circuit = QuantumCircuit(qr, cr)

        # Single‑qubit rotations
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)

        # Entanglement between adjacent qubits
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)

        circuit.measure(qr, cr)
        return circuit

    def run(self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            inputs: np.ndarray,
            shots: int = 1024) -> np.ndarray:
        """
        Execute the attention circuit for a batch of inputs.

        Parameters
        ----------
        rotation_params : np.ndarray
            Array of shape (3 * n_qubits,)
        entangle_params : np.ndarray
            Array of shape (n_qubits - 1,)
        inputs : np.ndarray
            Flattened input states for each query (shape: batch*seq_len, dim)
        shots : int
            Number of shots per circuit.

        Returns
        -------
        np.ndarray
            Probability matrix of shape (batch*seq_len, seq_len)
        """
        batch_size = inputs.shape[0]
        # For demonstration we generate a random probability distribution
        probs = np.random.rand(batch_size, self.seq_len)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs

class QLayer(tq.QuantumModule):
    """
    Quantum layer that implements a small variational circuit.
    """
    def __init__(self, n_wires: int):
        super().__init__()
        self.n_wires = n_wires
        # Encoder applies a parametrised RX on each wire
        self.encoder = tq.GeneralEncoder([
            {"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)
        ])
        self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
        self.encoder(qdev, x)
        for wire, gate in enumerate(self.params):
            gate(qdev, wires=wire)
        for wire in range(self.n_wires - 1):
            tqf.cnot(qdev, wires=[wire, wire + 1])
        return self.measure(qdev)

class QLSTM(nn.Module):
    """
    LSTM cell where each gate is realised by a QLayer.
    """
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

    def forward(self,
                inputs: torch.Tensor,
                states: tuple[torch.Tensor, torch.Tensor] | None = None):
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
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(self,
                     inputs: torch.Tensor,
                     states: tuple[torch.Tensor, torch.Tensor] | None = None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))

__all__ = ["QuantumSelfAttention", "QLSTM"]
